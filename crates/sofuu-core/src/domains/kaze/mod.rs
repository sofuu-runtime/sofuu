// domains/kaze/mod.rs — Plan 14 thin Rust adapter over `libsofuu_kaze.a`.
//
// All compilation (`check`, `compile`, `emit-c`, `to-machine`, `to-human`)
// calls the reentrant pure-Kaze compiler (`sofuu_ffi::kaze::Compiler`), and
// all bytecode execution (synchronous CLI/standalone and asynchronous QuickJS
// continuations) runs inside the resumable C99 VM (`sofuu_ffi::kaze::Vm`).

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock};

pub mod caps;
mod js_bridge;

pub use js_bridge::{cancel_async_runs_for_context, mod_sfs_register};

use caps::{CapHost, RuntimeHost, SimState};

pub const VERSION: &str = "1.0";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Allow(String),
    Reject(String),
}

#[derive(Clone, Debug)]
pub struct Limits {
    pub fuel: u64,
    pub heap_bytes: usize,
    pub arena_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            fuel: 100_000,
            heap_bytes: 4 * 1024 * 1024,
            arena_bytes: 256 * 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KazeError {
    pub code: String,
    pub message: String,
    pub location: Option<KazeLocation>,
}

#[derive(Debug, Clone)]
pub struct KazeLocation {
    pub source: String,
    pub line: u32,
    pub column: Option<u32>,
    pub routine_start: bool,
    pub bytecode_source: bool,
}

impl KazeError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        KazeError {
            code: code.to_string(),
            message: message.into(),
            location: None,
        }
    }

    pub fn at(
        code: &str,
        message: impl Into<String>,
        source: impl Into<String>,
        line: u32,
        column: Option<u32>,
    ) -> Self {
        KazeError {
            code: code.to_string(),
            message: message.into(),
            location: Some(KazeLocation {
                source: source.into(),
                line,
                column,
                routine_start: false,
                bytecode_source: false,
            }),
        }
    }

    pub fn render(&self) -> String {
        format!("{}: {}", self.code, self.message)
    }

    pub fn diagnostic(&self, source_text: Option<&str>) -> crate::rt::diag_contract::Diagnostic {
        let mut d = crate::rt::diag_contract::Diagnostic::from_code(&self.code, self.detail());
        if let Some(loc) = self.location.as_ref() {
            let mut location = crate::rt::diag_contract::SourceLocation::at(
                loc.source.clone(),
                loc.line,
                loc.column,
            );
            location.label = match (loc.routine_start, loc.bytecode_source) {
                (true, true) => Some(
                    "bytecode routine start; instruction line and original source unavailable"
                        .into(),
                ),
                (true, false) => Some("routine start; instruction line unavailable".into()),
                (false, true) => Some("source line stored in bytecode; original source unavailable".into()),
                (false, false) => None,
            };
            if let Some(src) = source_text {
                if loc.line > 0 {
                    if let Some(line) = src.lines().nth((loc.line - 1) as usize) {
                        location.snippet = Some(line.to_string());
                    }
                }
            }
            d = d.with_location(location);
        }
        d
    }

    fn detail(&self) -> String {
        let first = if self.code == "E_KAZE_CHECK" {
            self.message.lines().next().unwrap_or("")
        } else {
            self.message.as_str()
        };
        if let Some(loc) = self.location.as_ref() {
            let with_col = format!("{}:{}:{}: ", loc.source, loc.line, loc.column.unwrap_or(0));
            let line_only = format!("{}:{}: ", loc.source, loc.line);
            let source_only = format!("{}: ", loc.source);
            for prefix in [with_col, line_only, source_only] {
                if let Some(rest) = first.strip_prefix(&prefix) {
                    return strip_line_suffix(rest, loc.line);
                }
            }
        }
        strip_line_suffix(first, self.location.as_ref().map_or(0, |l| l.line))
    }
}

fn strip_line_suffix(msg: &str, line: u32) -> String {
    if line == 0 {
        return msg.to_string();
    }
    let suffix = format!(" (line {line})");
    msg.strip_suffix(&suffix).unwrap_or(msg).to_string()
}

#[derive(Debug)]
pub struct Prepared {
    pub bytecode: Vec<u8>,
    pub name: String,
    pub from_bytecode: bool,
    pub items: Vec<sofuu_ffi::kaze::ItemInfo>,
    pub needs: Vec<String>,
    pub cap_count: usize,
    pub const_count: usize,
    pub code_bytes: usize,
    pub line_map_empty: bool,
}

fn with_compiler<T>(f: impl FnOnce(&mut sofuu_ffi::kaze::Compiler) -> T) -> Result<T, KazeError> {
    struct SendCompiler(sofuu_ffi::kaze::Compiler);
    unsafe impl Send for SendCompiler {}

    static COMPILER: OnceLock<Mutex<SendCompiler>> = OnceLock::new();
    let lock = COMPILER.get_or_init(|| {
        let compiler = sofuu_ffi::kaze::Compiler::new(16 * 1024 * 1024)
            .expect("libsofuu_kaze compiler allocation failed");
        Mutex::new(SendCompiler(compiler))
    });
    let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    Ok(f(&mut guard.0))
}

fn parse_diag_line(line: &str) -> Option<(&str, u32, &str)> {
    let trimmed = line.trim();
    if let Some((head, msg)) = trimmed.split_once(": ") {
        let head = head.strip_prefix("compile: ").unwrap_or(head);
        if let Some((code, ln_str)) = head.split_once(" at line ") {
            if let Ok(ln) = ln_str.trim().parse::<u32>() {
                return Some((code.trim(), ln, msg));
            }
        }
        if let Some(ln_str) = head.strip_prefix("line ") {
            if let Ok(ln) = ln_str.trim().parse::<u32>() {
                return Some(("E_KAZE_CHECK", ln, msg));
            }
        }
    }
    None
}

fn compiler_error(
    name: &str,
    outcome: &sofuu_ffi::kaze::EmbedOutcome,
    default_code: &'static str,
) -> KazeError {
    let mut lines: Vec<String> = Vec::new();
    let mut first_code: Option<&str> = None;
    let mut first_line: u32 = outcome.err_line.max(0) as u32;
    for raw in outcome.text.lines() {
        let t = raw.trim();
        if t.is_empty()
            || t.starts_with("check: ")
            || t.starts_with("compile: ") && t.ends_with("no module written")
        {
            continue;
        }
        if let Some((code, ln, msg)) = parse_diag_line(t) {
            if first_code.is_none() {
                first_code = Some(code);
                if ln > 0 {
                    first_line = ln;
                }
            }
            lines.push(format!("{name}:{ln}: {msg}"));
        } else {
            lines.push(t.to_string());
        }
    }
    let body = if lines.is_empty() {
        outcome.text.trim().to_string()
    } else {
        lines.join("\n")
    };
    let is_parse = first_code == Some("E_KAZE_PARSE");
    let code = if is_parse {
        "E_KAZE_PARSE"
    } else {
        default_code
    };
    KazeError::at(
        code,
        body,
        name,
        first_line,
        if is_parse { Some(1) } else { None },
    )
}

fn decode_hex(hex: &str) -> Result<Vec<u8>, KazeError> {
    let clean = hex.trim();
    if clean.len() % 2 != 0 || clean.is_empty() {
        return Err(KazeError::new(
            "E_KAZE_BADCODE",
            "embedded compiler emitted malformed bytecode",
        ));
    }
    let bytes = clean.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let s = std::str::from_utf8(pair)
            .map_err(|_| KazeError::new("E_KAZE_BADCODE", "non-UTF8 compiler bytecode"))?;
        let b = u8::from_str_radix(s, 16)
            .map_err(|_| KazeError::new("E_KAZE_BADCODE", "invalid hex digit in bytecode"))?;
        out.push(b);
    }
    Ok(out)
}

fn bytecode_has_line_map(bytes: &[u8]) -> bool {
    if bytes.len() < 10 || &bytes[0..8] != b"KAZEBC1\0" {
        return false;
    }
    u16::from_le_bytes([bytes[8], bytes[9]]) == 2
}

pub fn load_bytecode(bytes: Vec<u8>, name: &str) -> Result<Prepared, KazeError> {
    let vm = sofuu_ffi::kaze::Vm::new(&bytes, 1024 * 1024, 1).ok_or_else(|| {
        KazeError::new("E_KAZE_BADCODE", sofuu_ffi::kaze::Vm::last_error())
    })?;
    let count = vm.item_count();
    let mut items = Vec::with_capacity(count);
    for i in 0..count {
        items.push(vm.item(i));
    }
    let needs = vm.needs();
    let cap_count = vm.cap_count();
    let const_count = vm.const_count();
    let code_bytes = vm.code_bytes();
    let line_map_empty = !bytecode_has_line_map(&bytes);
    Ok(Prepared {
        bytecode: bytes,
        name: name.to_string(),
        from_bytecode: true,
        items,
        needs,
        cap_count,
        const_count,
        code_bytes,
        line_map_empty,
    })
}

pub fn read_source(path: &str) -> Result<String, KazeError> {
    let bytes = std::fs::read(path)
        .map_err(|e| KazeError::new("E_KAZE_IO", format!("cannot read '{}': {}", path, e)))?;
    decode_source(bytes, path)
}

pub fn decode_source(bytes: Vec<u8>, path: &str) -> Result<String, KazeError> {
    let mut src = match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => {
            let at = e.utf8_error().valid_up_to();
            return Err(KazeError::new(
                "E_KAZE_PARSE",
                format!(
                    "'{}' is not valid UTF-8: the first bad byte is at offset {} (Kaze sources are UTF-8)",
                    path, at
                ),
            ));
        }
    };
    if let Some(rest) = src.strip_prefix('\u{feff}') {
        src = rest.to_string();
    }
    Ok(src)
}

pub fn check_source(src: &str, name: &str) -> Result<(), KazeError> {
    let outcome = with_compiler(|c| c.check(src))?;
    if !outcome.ok {
        return Err(compiler_error(name, &outcome, "E_KAZE_CHECK"));
    }
    Ok(())
}

pub fn prepare(src: &str, name: &str) -> Result<Prepared, KazeError> {
    check_source(src, name)?;
    let outcome = with_compiler(|c| c.compile(src))?;
    if !outcome.ok {
        return Err(compiler_error(name, &outcome, "E_KAZE_COMPILE"));
    }
    let bytes = decode_hex(&outcome.text)?;
    let mut prepared = load_bytecode(bytes, name)?;
    prepared.from_bytecode = false;
    Ok(prepared)
}

pub fn prepare_file(path: &str) -> Result<Prepared, KazeError> {
    let src = read_source(path)?;
    prepare(&src, path)
}

pub fn emit_c_source(src: &str, name: &str, standalone: bool) -> Result<String, KazeError> {
    check_source(src, name)?;
    let outcome = with_compiler(|c| c.emit_c(src, standalone))?;
    if !outcome.ok {
        return Err(compiler_error(name, &outcome, "E_KAZE_COMPILE"));
    }
    Ok(outcome.text)
}

pub fn to_machine(src: &str, name: &str) -> Result<String, KazeError> {
    let outcome = with_compiler(|c| c.to_machine(src))?;
    if !outcome.ok || outcome.text.contains("(unknown_item ") {
        let chk = with_compiler(|c| c.check(src))?;
        if !chk.ok {
            return Err(compiler_error(name, &chk, "E_KAZE_PARSE"));
        }
        return Err(KazeError::at(
            "E_KAZE_PARSE",
            format!("{name}:1: unable to convert source to machine syntax"),
            name,
            1,
            Some(1),
        ));
    }
    Ok(outcome.text)
}

pub fn to_human(src: &str, name: &str) -> Result<String, KazeError> {
    let outcome = with_compiler(|c| c.to_human(src))?;
    if !outcome.ok
        || outcome.text.trim().is_empty()
        || outcome.text.trim_start().starts_with("# E_KAZE_UNSUPPORTED:")
    {
        return Err(KazeError::at(
            "E_KAZE_PARSE",
            format!("{name}:1: invalid or unsupported machine S-expression"),
            name,
            1,
            Some(1),
        ));
    }
    Ok(outcome.text)
}

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub entry: Option<String>,
    pub args: serde_json::Map<String, serde_json::Value>,
    pub ambient_args: bool,
    pub config_path: Option<String>,
    pub grants: Vec<String>,
    pub fuel: u64,
    pub timeout_ms: u64,
    pub heap_bytes: usize,
    pub ticks: u32,
    pub tick_ms: u64,
    pub sim: SimState,
    pub policy: Option<serde_json::Value>,
    pub runtime_policy: Option<serde_json::Value>,
    pub backend: Vec<(String, String)>,
    pub pins: Vec<(String, u16)>,
    pub source_name: Option<String>,
}

impl Default for RunOptions {
    fn default() -> Self {
        let limits = Limits::default();
        RunOptions {
            entry: None,
            args: serde_json::Map::new(),
            ambient_args: false,
            config_path: None,
            grants: Vec::new(),
            fuel: limits.fuel,
            timeout_ms: 0,
            heap_bytes: limits.heap_bytes,
            ticks: 0,
            tick_ms: 100,
            sim: SimState::default(),
            policy: None,
            runtime_policy: None,
            backend: Vec::new(),
            pins: Vec::new(),
            source_name: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendSelection {
    pub family: String,
    pub backend: String,
    pub simulated: bool,
}

pub fn resolve_backends(
    opts: &RunOptions,
    source_name: &str,
) -> Result<Vec<BackendSelection>, KazeError> {
    let mut selection: Vec<BackendSelection> = Vec::new();
    for (family, backend) in &opts.backend {
        if let Err(d) = crate::rt::hardware::validate_grant(family, backend) {
            return Err(KazeError::new(
                d.code,
                format!("{}: {} ({})", source_name, d.reason, d.rule),
            ));
        }
        let resolved = caps::family_of(family).to_string();
        let simulated = backend == "sim";
        match selection.iter_mut().find(|s| s.family == resolved) {
            Some(existing) => {
                if !existing.backend.is_empty() && !backend.is_empty() && existing.backend != *backend {
                    return Err(KazeError::new(
                        "E_CAP_INVALID",
                        format!(
                            "{}: '{}' was granted two different backends ('{}' and '{}')",
                            source_name, resolved, existing.backend, backend
                        ),
                    ));
                }
                if existing.backend.is_empty() && !backend.is_empty() {
                    existing.backend = backend.clone();
                    existing.simulated = simulated;
                }
            }
            None => selection.push(BackendSelection {
                family: resolved,
                backend: backend.clone(),
                simulated,
            }),
        }
    }
    Ok(selection)
}

pub fn simulated_families(selection: &[BackendSelection]) -> Vec<String> {
    selection
        .iter()
        .filter(|s| s.simulated)
        .map(|s| s.family.clone())
        .collect()
}

pub fn blocked_live_families(selection: &[BackendSelection], pins: bool) -> Vec<String> {
    let mut blocked: Vec<String> = selection
        .iter()
        .filter(|s| {
            !s.backend.is_empty()
                && !s.simulated
                && matches!(s.family.as_str(), "drone" | "rover" | "sensor" | "device")
        })
        .map(|s| s.family.clone())
        .collect();
    if pins && !blocked.iter().any(|f| f == "device") {
        blocked.push("device".to_string());
    }
    blocked
}

#[derive(Debug, Clone)]
pub struct RunReport {
    pub entry: String,
    pub value: String,
    pub actions: Vec<String>,
    pub logs: Vec<(String, String)>,
    pub fuel_used: u64,
    pub instructions: u64,
    pub ticks: u32,
    pub verdict: Option<Verdict>,
    pub simulated_families: Vec<String>,
}

impl RunReport {
    pub fn render_plain(&self) -> String {
        let mut out = String::new();
        if !self.simulated_families.is_empty() {
            out.push_str(&format!(
                "simulation (testing): {}\n",
                self.simulated_families.join(", ")
            ));
        }
        if !self.entry.is_empty() {
            out.push_str(&format!("{} -> {}\n", self.entry, self.value));
        }
        if !self.actions.is_empty() {
            out.push_str("actions:\n");
            for (i, a) in self.actions.iter().enumerate() {
                out.push_str(&format!("  {:>2}. {}\n", i + 1, a));
            }
        }
        if let Some(v) = &self.verdict {
            match v {
                Verdict::Allow(text) => out.push_str(&format!("verdict: allow {}\n", text)),
                Verdict::Reject(reason) => out.push_str(&format!("verdict: reject - {}\n", reason)),
            }
        }
        out.push_str(&format!(
            "{} instructions, {} fuel, {} ticks\n",
            self.instructions, self.fuel_used, self.ticks
        ));
        out
    }
}

pub fn run_source(src: &str, name: &str, opts: &RunOptions) -> Result<RunReport, KazeError> {
    let prepared = prepare(src, name)?;
    run_prepared(prepared, opts)
}

pub fn run_file(path: &str, opts: &RunOptions) -> Result<RunReport, KazeError> {
    let prepared = prepare_file(path)?;
    run_prepared(prepared, opts)
}

pub fn run_prepared_source(prepared: Prepared, opts: &RunOptions) -> Result<RunReport, KazeError> {
    run_prepared(prepared, opts)
}

pub fn run_module(bytes: Vec<u8>, name: &str, opts: &RunOptions) -> Result<RunReport, KazeError> {
    let prepared = load_bytecode(bytes, name)?;
    run_prepared(prepared, opts)
}

pub(crate) fn build_cap_host(
    opts: &RunOptions,
    selection: &[BackendSelection],
    declared: &[String],
) -> Result<Box<dyn CapHost>, KazeError> {
    let has_live_drone = selection
        .iter()
        .any(|s| s.family == "drone" && !s.backend.is_empty() && !s.simulated);
    if has_live_drone || !opts.pins.is_empty() {
        let backends: Vec<(String, String)> = selection
            .iter()
            .map(|s| (s.family.clone(), s.backend.clone()))
            .collect();
        return crate::domains::robotics::kaze_live::LiveHost::new(
            opts.sim.clone(),
            &backends,
            &opts.pins,
        )
        .map(|h| Box::new(h) as Box<dyn CapHost>)
        .map_err(|msg| {
            if let Some((code, rest)) = msg.split_once(": ") {
                KazeError::new(code, rest)
            } else {
                KazeError::new("E_KAZE_LINK", msg)
            }
        });
    }
    let alert_sink = selection
        .iter()
        .find(|s| s.family == "alert" && !s.backend.is_empty())
        .map(|s| s.backend.clone());
    let base_host = RuntimeHost::with_alert(
        opts.sim.clone(),
        simulated_families(selection),
        blocked_live_families(selection, !opts.pins.is_empty()),
        alert_sink,
    );
    let declares_memory = declared.iter().any(|d| caps::family_of(d) == "memory");
    let memory_backend = selection
        .iter()
        .find(|s| s.family == "memory")
        .map(|s| s.backend.as_str())
        .filter(|_| declares_memory);
    match memory_backend {
        Some("ephemeral") => Ok(Box::new(crate::rt::kaze_memory::EphemeralMemoryHost::new(
            base_host,
        ))),
        Some("durable") => {
            let default_path = std::path::PathBuf::from(".sofuu/kaze_memory.json");
            Ok(Box::new(
                crate::rt::kaze_memory::DurableMemoryHost::new(base_host, default_path)
                    .map_err(|e| KazeError::new("E_CONFIG", e))?,
            ))
        }
        Some(b) if b.starts_with("file:") || b.starts_with("path:") => {
            let path = b[5..].trim();
            if path.is_empty() {
                return Err(KazeError::new(
                    "E_CONFIG",
                    "a durable Kaze memory store needs a file path after 'file:' or 'path:'",
                ));
            }
            Ok(Box::new(
                crate::rt::kaze_memory::DurableMemoryHost::new(
                    base_host,
                    std::path::PathBuf::from(path),
                )
                .map_err(|e| KazeError::new("E_CONFIG", e))?,
            ))
        }
        _ => Ok(Box::new(base_host)),
    }
}

pub(crate) struct CShadow {
    pub(crate) host: Box<dyn CapHost>,
    pub(crate) declared: Vec<String>,
    pub(crate) grants: Vec<String>,
    pub(crate) policy: Option<serde_json::Value>,
    pub(crate) runtime_policy: Option<serde_json::Value>,
    pub(crate) async_mode: bool,
}

fn c_text(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

unsafe fn c_write(dst: *mut c_char, cap: usize, text: &str) -> usize {
    if dst.is_null() || cap == 0 {
        return 0;
    }
    let bytes = text.as_bytes();
    let n = bytes.len().min(cap - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst as *mut u8, n);
        *dst.add(n) = 0;
    }
    n
}

pub(crate) extern "C" fn c_authorise(
    ctx: *mut c_void,
    family: *const c_char,
    verb: *const c_char,
    args_json: *const c_char,
    out_err: *mut c_char,
    err_cap: usize,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> c_int {
        let shadow = unsafe { &mut *(ctx as *mut CShadow) };
        let family = c_text(family);
        let verb = c_text(verb);
        let args: Vec<serde_json::Value> = match serde_json::from_str(&c_text(args_json)) {
            Ok(args) => args,
            Err(e) => {
                unsafe {
                    c_write(
                        out_err,
                        err_cap,
                        &format!("E_KAZE_HOST: capability arguments are not valid JSON: {e}"),
                    )
                };
                return 0;
            }
        };
        let denied = crate::rt::hardware::authorise(
            &shadow.declared,
            &shadow.grants,
            shadow.policy.as_ref(),
            &family,
            &verb,
        )
        .err()
        .or_else(|| {
            shadow.runtime_policy.as_ref().and_then(|policy| {
                crate::rt::hardware::authorise_runtime_policy(policy, &family, &verb).err()
            })
        })
        .or_else(|| crate::rt::hardware::validate_arguments(&family, &verb, &args).err());
        match denied {
            None => 1,
            Some(d) => {
                unsafe {
                    c_write(out_err, err_cap, &format!("{}: {} ({})", d.code, d.reason, d.rule))
                };
                0
            }
        }
    }));
    match result {
        Ok(code) => code,
        Err(_) => {
            unsafe { c_write(out_err, err_cap, "E_KAZE_HOST: the capability gate panicked") };
            0
        }
    }
}

pub(crate) extern "C" fn c_dispatch(
    ctx: *mut c_void,
    family: *const c_char,
    verb: *const c_char,
    args_json: *const c_char,
    out_json: *mut c_char,
    out_cap: usize,
    out_len: *mut usize,
    out_err: *mut c_char,
    err_cap: usize,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> c_int {
        let shadow = unsafe { &mut *(ctx as *mut CShadow) };
        let family = c_text(family);
        let verb = c_text(verb);
        if shadow.async_mode && matches!(family.as_str(), "ai" | "tool") {
            return sofuu_ffi::kaze::CAP_ASYNC;
        }
        let args: Vec<serde_json::Value> = match serde_json::from_str(&c_text(args_json)) {
            Ok(args) => args,
            Err(e) => {
                unsafe {
                    c_write(
                        out_err,
                        err_cap,
                        &format!("E_KAZE_HOST: capability arguments are not valid JSON: {e}"),
                    )
                };
                return sofuu_ffi::kaze::CAP_FAILED;
            }
        };
        match shadow.host.call(&family, &verb, &args) {
            Ok(json) => {
                let text = match serde_json::to_string(&json) {
                    Ok(text) => text,
                    Err(_) => {
                        unsafe {
                            c_write(
                                out_err,
                                err_cap,
                                "E_KAZE_HOST: capability result is not valid JSON",
                            )
                        };
                        return sofuu_ffi::kaze::CAP_FAILED;
                    }
                };
                let n = unsafe { c_write(out_json, out_cap, &text) };
                if !out_len.is_null() {
                    unsafe { *out_len = n };
                }
                sofuu_ffi::kaze::CAP_OK
            }
            Err(message) => {
                unsafe { c_write(out_err, err_cap, &message) };
                sofuu_ffi::kaze::CAP_FAILED
            }
        }
    }));
    match result {
        Ok(code) => code,
        Err(_) => {
            unsafe { c_write(out_err, err_cap, "E_KAZE_HOST: the capability dispatcher panicked") };
            sofuu_ffi::kaze::CAP_FAILED
        }
    }
}

pub(crate) extern "C" fn c_poll_deadline(_ctx: *mut c_void) -> c_int {
    if crate::rt::deadline::poll_expired() != 0 {
        1
    } else {
        0
    }
}

pub(crate) fn bind_args_json(
    params: &[String],
    args: &serde_json::Map<String, serde_json::Value>,
) -> Result<String, KazeError> {
    let missing: Vec<&String> = params
        .iter()
        .enumerate()
        .filter(|(i, p)| args.get(*p).is_none() && args.get(&i.to_string()).is_none())
        .map(|(_, p)| p)
        .collect();
    if !missing.is_empty() {
        return Err(KazeError::new(
            "E_KAZE_PARAM",
            missing_param_message(&missing),
        ));
    }
    let values: Vec<&serde_json::Value> = params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            args.get(p)
                .or_else(|| args.get(&i.to_string()))
                .expect("checked above")
        })
        .collect();
    serde_json::to_string(&values).map_err(|e| KazeError::new("E_KAZE_ARGS", e.to_string()))
}

pub(crate) fn c_item_error(
    name: &str,
    error: &sofuu_ffi::kaze::Vm,
    from_bytecode: bool,
    item_line: u32,
    line_map_empty: bool,
) -> KazeError {
    let line = error.error_line();
    let mut err = KazeError::at(
        vm_code_static(&error.error_code()),
        format!("{}: {} (line {})", name, error.error_message(), line),
        name,
        line,
        None,
    );
    if let Some(loc) = &mut err.location {
        loc.routine_start = line_map_empty && line != 0 && line == item_line;
        loc.bytecode_source = from_bytecode;
    }
    err
}

fn vm_code_static(code: &str) -> &'static str {
    for candidate in [
        "E_CAP_MISSING",
        "E_CAP_DENIED",
        "E_CAP_UNDECLARED",
        "E_CAP_INVALID",
        "E_CAP_APPROVAL_REQUIRED",
        "E_KAZE_MEMORY",
        "E_KAZE_RANGE",
        "E_KAZE_TYPE",
        "E_KAZE_PARAM",
        "E_KAZE_BADCODE",
        "E_KAZE_OVERFLOW",
        "E_KAZE_DIV0",
        "E_KAZE_BOUNDS",
        "E_KAZE_LINK",
        "E_KAZE_LINK_TIMEOUT",
        "E_COMMAND_REJECTED",
        "E_ARMING_DENIED",
        "E_SAFETY_VIOLATION",
        "E_KAZE_HOST",
        "E_KAZE_STATE",
        "E_KAZE_TRAP",
        "E_KAZE_ENTRY",
        "E_TIMEOUT",
    ] {
        if code == candidate {
            return candidate;
        }
    }
    "E_KAZE_TRAP"
}

fn run_prepared(prepared: Prepared, opts: &RunOptions) -> Result<RunReport, KazeError> {
    let declared = prepared.needs.clone();
    let selection = resolve_backends(opts, &prepared.name)?;
    let host = build_cap_host(opts, &selection, &declared)?;
    let limits = Limits {
        fuel: opts.fuel.max(1),
        heap_bytes: opts.heap_bytes,
        ..Limits::default()
    };

    let entry = resolve_entry(&prepared, opts)?;
    if entry.is_none() {
        layered_args(opts, &[])?;
    }

    let mut report = RunReport {
        entry: entry.clone().unwrap_or_default(),
        value: "nothing".to_string(),
        actions: Vec::new(),
        logs: Vec::new(),
        fuel_used: 0,
        instructions: 0,
        ticks: 0,
        verdict: None,
        simulated_families: simulated_families(&selection),
    };

    let mut shadow = CShadow {
        host,
        declared,
        grants: opts.grants.clone(),
        policy: opts.policy.clone(),
        runtime_policy: opts.runtime_policy.clone(),
        async_mode: false,
    };
    let cbs = sofuu_ffi::kaze::HostCbs {
        ctx: (&mut shadow as *mut CShadow).cast::<c_void>(),
        cap_dispatch: Some(c_dispatch),
        authorise: Some(c_authorise),
        poll_deadline: Some(c_poll_deadline),
        ..sofuu_ffi::kaze::HostCbs::default()
    };

    let mut vm = sofuu_ffi::kaze::Vm::new(
        &prepared.bytecode,
        limits.arena_bytes.max(1024 * 1024),
        limits.fuel,
    )
    .ok_or_else(|| {
        KazeError::new(
            "E_KAZE_MEMORY",
            format!("cannot create the Kaze VM: {}", sofuu_ffi::kaze::Vm::last_error()),
        )
    })?;
    vm.set_heap_limit(limits.heap_bytes);
    vm.set_host(&cbs);

    let line_map_empty = prepared.line_map_empty;
    let run_item = |vm: &mut sofuu_ffi::kaze::Vm, item_idx: usize, item_line: u32, args: Option<&str>| -> Result<(), KazeError> {
        if !vm.start(item_idx, args) {
            return Err(c_item_error(
                &prepared.name,
                vm,
                prepared.from_bytecode,
                item_line,
                line_map_empty,
            ));
        }
        loop {
            let (status, _) = vm.step(256);
            match status {
                sofuu_ffi::kaze::SLICE_YIELD => continue,
                sofuu_ffi::kaze::SLICE_DONE => return Ok(()),
                sofuu_ffi::kaze::SLICE_CAP_REQ => {
                    let (line, family, verb, _) = vm.pending();
                    return Err(KazeError::at(
                        "E_CAP_MISSING",
                        format!(
                            "{}: {}.{} requires an asynchronous capability driver (line {})",
                            prepared.name, family, verb, line
                        ),
                        &prepared.name,
                        line,
                        None,
                    ));
                }
                sofuu_ffi::kaze::SLICE_ERR => {
                    return Err(c_item_error(
                        &prepared.name,
                        vm,
                        prepared.from_bytecode,
                        item_line,
                        line_map_empty,
                    ));
                }
                _ => {
                    return Err(KazeError::new(
                        "E_KAZE_STATE",
                        "the Kaze VM returned an unknown slice status",
                    ))
                }
            }
        }
    };

    let mut entry_json = "null".to_string();
    if let Some(name) = &entry {
        let (idx, item) = prepared
            .items
            .iter()
            .enumerate()
            .find(|(_, i)| &i.name == name)
            .ok_or_else(|| {
                KazeError::new("E_KAZE_ENTRY", format!("there is no '{}' in this script", name))
            })?;
        let merged = layered_args(opts, &item.params)?;
        let args = bind_args_json(&item.params, &merged)?;
        run_item(&mut vm, idx, item.line, Some(&args))?;
        entry_json = vm.value_json();
    }

    for tick in 0..opts.ticks {
        let t_ms = tick as f64 * opts.tick_ms as f64;
        for (idx, item) in prepared.items.iter().enumerate() {
            let due = match item.tag.as_str() {
                "when" => true,
                "every" => item.period_ms > 0.0 && (t_ms % item.period_ms) == 0.0,
                _ => false,
            };
            if !due {
                continue;
            }
            run_item(&mut vm, idx, item.line, None)?;
        }
        report.ticks += 1;
    }

    report.fuel_used = vm.fuel_used();
    report.instructions = vm.instructions();
    report.verdict = match vm.verdict_kind().as_deref() {
        Some("allow") => Some(Verdict::Allow(vm.verdict_text())),
        Some("reject") => Some(Verdict::Reject(vm.verdict_text())),
        _ => None,
    };
    let val: serde_json::Value = serde_json::from_str(&entry_json).map_err(|e| {
        KazeError::new(
            "E_KAZE_MEMORY",
            format!("the Kaze VM returned an unreadable value: {e}"),
        )
    })?;
    report.value = caps::render_value(&val);
    report.actions = shadow.host.actions().iter().map(|a| a.render()).collect();
    report.logs = shadow.host.logs().to_vec();
    Ok(report)
}

pub(crate) fn resolve_entry(
    prepared: &Prepared,
    opts: &RunOptions,
) -> Result<Option<String>, KazeError> {
    if let Some(name) = &opts.entry {
        return Ok(Some(name.clone()));
    }
    if prepared.items.iter().any(|i| i.name == "main") {
        return Ok(Some("main".to_string()));
    }
    let routines: Vec<&sofuu_ffi::kaze::ItemInfo> = prepared
        .items
        .iter()
        .filter(|i| matches!(i.tag.as_str(), "action" | "check" | "filter"))
        .collect();
    if routines.len() == 1 {
        return Ok(Some(routines[0].name.clone()));
    }
    if routines.is_empty() && !prepared.items.is_empty() {
        return Ok(None);
    }
    Err(KazeError::new(
        "E_KAZE_ENTRY",
        "this script has more than one routine; name the one to run with --entry (or add an action main())",
    ))
}

fn missing_param_message(missing: &[&String]) -> String {
    let names: Vec<String> = missing.iter().map(|n| format!("'{}'", n)).collect();
    let plural = missing.len() > 1;
    let first = missing[0];
    let env_name = first.to_ascii_uppercase().replace('-', "_");
    format!(
        "missing required parameter{s} {list}. Supply via:\n           1. CLI flag: --arg {first}=<value> or positional argument\n           2. Environment variable: AGENT_{env_name}=<value>\n           3. Configuration file: --config <path> or ./agent.config.json",
        s = if plural { "s" } else { "" },
        list = names.join(", "),
    )
}

pub(crate) fn layered_args(
    opts: &RunOptions,
    params: &[String],
) -> Result<serde_json::Map<String, serde_json::Value>, KazeError> {
    if !opts.ambient_args {
        return Ok(opts.args.clone());
    }
    let mut out = serde_json::Map::new();
    let config_path = opts.config_path.as_deref().unwrap_or("agent.config.json");
    match std::fs::read_to_string(config_path) {
        Ok(text) => {
            let parsed: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
                KazeError::new(
                    "E_KAZE_ARGS",
                    format!("invalid configuration '{}': {}", config_path, e),
                )
            })?;
            let mut cfg = parsed.as_object().cloned().ok_or_else(|| {
                KazeError::new(
                    "E_KAZE_ARGS",
                    format!("configuration '{}' must be an object", config_path),
                )
            })?;
            let map = match cfg.remove("args") {
                Some(serde_json::Value::Object(args)) => args,
                Some(_) => {
                    return Err(KazeError::new(
                        "E_KAZE_ARGS",
                        format!("configuration '{}' args must be an object", config_path),
                    ))
                }
                None => cfg,
            };
            for (k, v) in map {
                out.insert(k, v);
            }
        }
        Err(e) if opts.config_path.is_none() && e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(KazeError::new(
                "E_KAZE_IO",
                format!("cannot read configuration '{}': {}", config_path, e),
            ))
        }
    }
    for p in params {
        let env_name = p.to_ascii_uppercase().replace('-', "_");
        for key in [format!("AGENT_{env_name}"), format!("SOFUU_{env_name}")] {
            if let Ok(raw) = std::env::var(&key) {
                let v = serde_json::from_str(&raw).unwrap_or(serde_json::Value::String(raw));
                out.insert(p.clone(), v);
                break;
            }
        }
    }
    for (i, p) in params.iter().enumerate() {
        if let Some(v) = opts.args.get(&i.to_string()) {
            out.insert(p.clone(), v.clone());
        }
    }
    for (k, v) in &opts.args {
        out.insert(k.clone(), v.clone());
    }
    Ok(out)
}

pub struct KazeSubsystem;

impl crate::domains::Subsystem for KazeSubsystem {
    fn name(&self) -> &'static str {
        "kaze"
    }

    fn register_js(&self, ctx: *mut sofuu_ffi::qjs::JSContext) -> Result<(), &'static str> {
        unsafe {
            mod_sfs_register(ctx);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pure_program_runs_via_embedded_c_compiler_and_vm() {
        let report =
            run_source("action main():\n    return 2 + 3\n", "<pure>", &RunOptions::default())
                .unwrap();
        assert_eq!(report.value, "5");
    }

    #[test]
    fn js_opts_reject_conflicts_and_ambient_args_stay_disabled_by_default() {
        let quiet = js_bridge::opts_from_json(None).unwrap();
        assert!(!quiet.ambient_args);
        let asking = js_bridge::opts_from_json(Some(json!({
            "ambientArgs": true,
            "configPath": "agent.config.json",
            "sourceName": "agent.kaze",
            "caps": {"drone": "sim"},
            "policy": {"tools": {"deny": []}},
        })))
        .unwrap();
        assert!(asking.ambient_args);
        assert_eq!(asking.config_path.as_deref(), Some("agent.config.json"));
        assert!(js_bridge::opts_from_json(Some(json!({"ambientArgs": "yes"}))).is_err());
    }

    #[test]
    fn machine_and_human_translation_run_through_embedded_compiler() {
        let src = "needs: log\n\naction main():\n    return 41 + 1\n";
        let m = to_machine(src, "m.kaze").unwrap();
        assert!(m.contains("(kaze 2"), "{m}");
        let h = to_human(&m, "m.kaze").unwrap();
        let report = run_source(&h, "h.kaze", &RunOptions::default()).unwrap();
        assert_eq!(report.value, "42");
    }
}
