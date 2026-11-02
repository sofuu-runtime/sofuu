// sfs/mod.rs — the Kaze engine: source → checker → bytecode → VM, and `sofuu.sfs`.
//
// PLAN-KAZE-LANGUAGE §5/§6. This module is the only public door: the CLI
// (`sofuu script …`) and the JS surface (`sofuu.sfs.run`) both come through
// `run_source`/`run_file`, so both get the same checker, the same fuel meter
// and the same R7 gate.
//
// Kaze is additive. QuickJS and every shipped driver are untouched; this is a
// separate engine for the work a general-purpose runtime is bad at —
// deterministic, resource-bounded, capability-gated behaviour.

pub mod arena;
pub mod ast;
pub mod caps;
pub mod checker;
pub mod compiler;
pub mod isa;
pub mod lexer;
pub mod parser;
pub mod value;
pub mod vm;

use std::ffi::{c_int, CString};

use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

use crate::sfs::caps::{Action, CapHost, SimHost, SimState};
use crate::sfs::isa::Module;
use crate::sfs::vm::{Limits, Vm, Verdict};

/// Language version reported by `sofuu.sfs.version` and `sofuu script --version`.
pub const VERSION: &str = "0.1";

#[derive(Debug, Clone)]
pub struct KazeError {
    pub code: String,
    pub message: String,
}

impl KazeError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        KazeError { code: code.to_string(), message: message.into() }
    }

    pub fn render(&self) -> String {
        format!("{}: {}", self.code, self.message)
    }
}

/// A compiled script plus the source name it came from.
pub struct Prepared {
    pub module: Module,
    pub name: String,
}

/// Read a script from disk.
pub fn read_source(path: &str) -> Result<String, KazeError> {
    std::fs::read_to_string(path)
        .map_err(|e| KazeError::new("E_KAZE_IO", format!("cannot read '{}': {}", path, e)))
}

/// Parse, check and compile. This is the whole front end in one call.
pub fn prepare(src: &str, name: &str) -> Result<Prepared, KazeError> {
    let program = parser::parse(src).map_err(|e| {
        KazeError::new("E_KAZE_PARSE", e.render(name))
    })?;
    checker::check(&program).map_err(|errors| {
        let mut message = String::new();
        for (i, err) in errors.iter().enumerate() {
            if i > 0 {
                message.push_str("\n");
            }
            message.push_str(&err.render(name));
        }
        KazeError::new("E_KAZE_CHECK", message)
    })?;
    let module = compiler::compile(&program)
        .map_err(|e| KazeError::new("E_KAZE_COMPILE", e.render(name)))?;
    Ok(Prepared { module, name: name.to_string() })
}

pub fn prepare_file(path: &str) -> Result<Prepared, KazeError> {
    let src = read_source(path)?;
    prepare(&src, path)
}

/// Front end only — what `sofuu script check` runs.
pub fn check_source(src: &str, name: &str) -> Result<(), KazeError> {
    prepare(src, name).map(|_| ())
}

/// Everything a run needs besides the script itself.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Item to run first. Default: `main`, else the only routine in the file.
    pub entry: Option<String>,
    pub args: serde_json::Map<String, serde_json::Value>,
    /// Capability families the host grants a backend for (`drone=sim` → "drone").
    pub grants: Vec<String>,
    pub fuel: u64,
    /// Budget for arrays and maps, in bytes. The second meter: `fuel` bounds
    /// instructions, this bounds retained memory.
    pub heap_bytes: usize,
    /// Supervisor ticks after the entry finishes; each tick polls `when`/`every`.
    pub ticks: u32,
    pub tick_ms: u64,
    pub sim: SimState,
    /// R7 policy layered over the hardware gate. `None` means "declaration and
    /// grant only", which is what the tests use.
    pub policy: Option<serde_json::Value>,
    /// `device name -> GPIO line` (`--pin fan=17`). Non-empty puts the real
    /// pins behind the `device` family (Kaze v0.2, P-K1).
    pub pins: Vec<(String, u16)>,
    /// `family -> device`. A device other than `sim` (for example
    /// `drone -> udp:127.0.0.1:14550`) puts the live backend behind that family
    /// (Kaze v0.2, P-K1). Empty means "everything simulated".
    pub backend: Vec<(String, String)>,
}

impl Default for RunOptions {
    fn default() -> Self {
        RunOptions {
            entry: None,
            args: serde_json::Map::new(),
            grants: Vec::new(),
            fuel: Limits::default().fuel,
            heap_bytes: value::Heap::DEFAULT_LIMIT,
            ticks: 0,
            tick_ms: 100,
            sim: SimState::default(),
            policy: None,
            backend: Vec::new(),
            pins: Vec::new(),
        }
    }
}

/// What one run did. Everything here is evidence: the trace, the numbers and
/// the verdict are all derived, not narrated.
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
}

impl RunReport {
    /// The canonical text form of a run — the line a mission review reads and
    /// the exact bytes the golden corpus stores. The CLI adds colour; nothing
    /// else about the shape changes.
    pub fn render_plain(&self) -> String {
        let mut out = String::new();
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

/// Run a script from source text.
pub fn run_source(src: &str, name: &str, opts: &RunOptions) -> Result<RunReport, KazeError> {
    let prepared = prepare(src, name)?;
    run_prepared(prepared, opts)
}

pub fn run_file(path: &str, opts: &RunOptions) -> Result<RunReport, KazeError> {
    let prepared = prepare_file(path)?;
    run_prepared(prepared, opts)
}

/// Run a compiled module directly: no lexer, no parser, no checker. This is the
/// path a `.kazebc` artifact takes, so a deployed script needs no front end.
pub fn run_module(module: Module, name: &str, opts: &RunOptions) -> Result<RunReport, KazeError> {
    run_prepared(Prepared { module, name: name.to_string() }, opts)
}

fn run_prepared(prepared: Prepared, opts: &RunOptions) -> Result<RunReport, KazeError> {
    let module = prepared.module;
    let declared = module.needs.clone();
    // v0.2 (P-K1): a family granted a real device runs live; every other
    // family keeps the deterministic simulator.
    let live = opts.backend.iter().any(|(_, d)| !d.is_empty() && d != "sim") || !opts.pins.is_empty();
    let mut host: Box<dyn CapHost> = if live {
        match crate::rt::kaze_live::LiveHost::new(opts.sim.clone(), &opts.backend, &opts.pins) {
            Ok(h) => Box::new(h),
            Err(e) => return Err(KazeError::new("E_KAZE_LINK", format!("{}: {}", prepared.name, e))),
        }
    } else {
        Box::new(SimHost::new(opts.sim.clone()))
    };
    let limits = Limits { fuel: opts.fuel.max(1), heap_bytes: opts.heap_bytes, ..Limits::default() };

    let entry = resolve_entry(&module, opts)?;

    let mut report = RunReport {
        entry: entry.clone().unwrap_or_default(),
        value: "nothing".to_string(),
        actions: Vec::new(),
        logs: Vec::new(),
        fuel_used: 0,
        instructions: 0,
        ticks: 0,
        verdict: None,
    };

    {
        let mut vm = Vm::new(&module, &mut *host, &declared, &opts.grants, opts.policy.as_ref(), limits);

        if let Some(name) = &entry {
            let item = module
                .item(name)
                .ok_or_else(|| KazeError::new("E_KAZE_ENTRY", format!("there is no '{}' in this script", name)))?;
            let args = bind_args(item.params.as_slice(), &opts.args, &mut vm)
                .map_err(|e| KazeError::new("E_KAZE_MEMORY", format!("{}: {}", prepared.name, e)))?;
            let value = vm
                .run_item(item, &args)
                .map_err(|e| KazeError::new(e.code, format!("{}: {} (line {})", prepared.name, e.msg, e.line)))?;
            report.value = caps::render_value(value, &vm.arena, &vm.heap);
            report.verdict = vm.take_verdict();
        }

        // Supervisory ticks: `when` predicates and due `every` blocks.
        for tick in 0..opts.ticks {
            let t_ms = tick as f64 * opts.tick_ms as f64;
            for item in module.items.iter() {
                let due = match item.tag {
                    isa::ItemTag::When => true,
                    isa::ItemTag::Every => item.period_ms > 0.0 && (t_ms % item.period_ms) == 0.0,
                    _ => false,
                };
                if !due {
                    continue;
                }
                vm.run_item(item, &[]).map_err(|e| {
                    KazeError::new(e.code, format!("{}: {} (line {})", prepared.name, e.msg, e.line))
                })?;
            }
            report.ticks += 1;
        }

        report.fuel_used = vm.fuel_used;
        report.instructions = vm.instructions;
    }

    report.actions = host.actions().iter().map(|a: &Action| a.render()).collect();
    report.logs = host.logs().to_vec();
    Ok(report)
}

fn resolve_entry(module: &Module, opts: &RunOptions) -> Result<Option<String>, KazeError> {
    if let Some(name) = &opts.entry {
        return Ok(Some(name.clone()));
    }
    if module.item("main").is_some() {
        return Ok(Some("main".to_string()));
    }
    let routines: Vec<&isa::ItemCode> = module
        .items
        .iter()
        .filter(|i| matches!(i.tag, isa::ItemTag::Action | isa::ItemTag::Check | isa::ItemTag::Filter))
        .collect();
    if routines.len() == 1 {
        return Ok(Some(routines[0].name.clone()));
    }
    if routines.is_empty() && !module.items.is_empty() {
        // Hooks only: the ticks are the program.
        return Ok(None);
    }
    Err(KazeError::new(
        "E_KAZE_ENTRY",
        "this script has more than one routine; name the one to run with --entry (or add an action main())",
    ))
}

fn bind_args(
    params: &[String],
    args: &serde_json::Map<String, serde_json::Value>,
    vm: &mut Vm,
) -> Result<Vec<value::Value>, String> {
    let mut out = Vec::with_capacity(params.len());
    for (i, p) in params.iter().enumerate() {
        let json = args
            .get(p)
            .or_else(|| args.get(&i.to_string()))
            .unwrap_or(&serde_json::Value::Null);
        out.push(json_to_value(json, &mut vm.arena, &mut vm.heap)?);
    }
    Ok(out)
}

fn json_to_value(
    j: &serde_json::Value,
    arena: &mut arena::Arena,
    heap: &mut value::Heap,
) -> Result<value::Value, String> {
    use serde_json::Value as J;
    Ok(match j {
        J::Null => value::Value::Unit,
        J::Bool(b) => value::Value::Bool(*b),
        J::Number(n) => value::Value::Num(n.as_f64().unwrap_or(0.0)),
        J::String(s) => match arena.alloc(s) {
            Ok(r) => value::Value::Str(r),
            Err(_) => value::Value::Unit,
        },
        J::Array(items) => {
            let mut vals = Vec::with_capacity(items.len());
            for i in items {
                vals.push(json_to_value(i, arena, heap)?);
            }
            heap.push_array(vals)?
        }
        J::Object(map) => {
            let mut entries = Vec::with_capacity(map.len());
            for (k, v) in map {
                let key = match arena.alloc(k) {
                    Ok(r) => value::Value::Str(r),
                    Err(_) => value::Value::Unit,
                };
                entries.push((key, json_to_value(v, arena, heap)?));
            }
            heap.push_map(entries)?
        }
    })
}

// ── the JS surface: sofuu.sfs ───────────────────────────────────────
//
//   sofuu.sfs.run(pathOrSource, args?, opts?)  -> report object
//   sofuu.sfs.check(pathOrSource)              -> { ok, errors }
//   sofuu.sfs.compile(pathOrSource)            -> { ok, bytes, items, caps, consts }
//   sofuu.sfs.version                          -> "0.1"
//
// A report always comes back (no throw): a tool loop can branch on `ok`
// without wrapping every call, and a failure still carries its code.

unsafe fn js_text(ctx: *mut JSContext, v: JSValueConst) -> String {
    let p = unsafe { qjs::sofuu_js_to_cstring(ctx, v) };
    if p.is_null() {
        return String::new();
    }
    let s = unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned();
    unsafe { qjs::sofuu_js_free_cstring(ctx, p) };
    s
}

unsafe fn js_json(ctx: *mut JSContext, v: JSValueConst) -> Option<serde_json::Value> {
    let text = {
        let j = unsafe { qjs::JS_JSONStringify(ctx, v, qjs::sofuu_js_undefined(), qjs::sofuu_js_undefined()) };
        if unsafe { qjs::is_exception(j) } {
            let exc = unsafe { qjs::JS_GetException(ctx) };
            unsafe { qjs::sofuu_js_free_value(ctx, exc) };
            return None;
        }
        let s = unsafe { js_text(ctx, j) };
        unsafe { qjs::sofuu_js_free_value(ctx, j) };
        s
    };
    serde_json::from_str(&text).ok()
}

/// A script argument is a path, or the script itself when it holds a newline.
fn source_of(text: &str) -> Result<(String, String), KazeError> {
    let looks_like_path = !text.contains('\n')
        && (text.ends_with(".kaze") || text.ends_with(".kz") || std::path::Path::new(text).exists());
    if looks_like_path {
        let src = read_source(text)?;
        Ok((src, text.to_string()))
    } else {
        Ok((text.to_string(), "<inline>".to_string()))
    }
}

fn report_to_js(ctx: *mut JSContext, report: &RunReport) -> JSValue {
    unsafe {
        let obj = qjs::sofuu_js_new_object(ctx);
        set_bool(ctx, obj, c"ok", true);
        set_str(ctx, obj, c"entry", &report.entry);
        set_str(ctx, obj, c"result", &report.value);
        set_num(ctx, obj, c"fuel", report.fuel_used as f64);
        set_num(ctx, obj, c"instructions", report.instructions as f64);
        set_num(ctx, obj, c"ticks", report.ticks as f64);
        let actions = qjs::JS_NewArray(ctx);
        for (i, a) in report.actions.iter().enumerate() {
            let v = qjs::sofuu_js_new_string(ctx, CString::new(a.as_str()).unwrap_or_default().as_ptr());
            qjs::JS_SetPropertyUint32(ctx, actions, i as u32, v);
        }
        qjs::sofuu_js_set_property_str(ctx, obj, c"actions".as_ptr(), actions);
        let logs = qjs::JS_NewArray(ctx);
        for (i, (level, msg)) in report.logs.iter().enumerate() {
            let entry = qjs::sofuu_js_new_object(ctx);
            set_str(ctx, entry, c"level", level);
            set_str(ctx, entry, c"message", msg);
            qjs::JS_SetPropertyUint32(ctx, logs, i as u32, entry);
        }
        qjs::sofuu_js_set_property_str(ctx, obj, c"logs".as_ptr(), logs);
        if let Some(v) = &report.verdict {
            let verdict = qjs::sofuu_js_new_object(ctx);
            match v {
                Verdict::Allow(text) => {
                    set_bool(ctx, verdict, c"allow", true);
                    set_str(ctx, verdict, c"reason", text);
                }
                Verdict::Reject(reason) => {
                    set_bool(ctx, verdict, c"allow", false);
                    set_str(ctx, verdict, c"reason", reason);
                }
            }
            qjs::sofuu_js_set_property_str(ctx, obj, c"verdict".as_ptr(), verdict);
        }
        obj
    }
}

fn error_to_js(ctx: *mut JSContext, err: &KazeError) -> JSValue {
    unsafe {
        let obj = qjs::sofuu_js_new_object(ctx);
        set_bool(ctx, obj, c"ok", false);
        set_str(ctx, obj, c"code", &err.code);
        set_str(ctx, obj, c"error", &err.message);
        obj
    }
}

unsafe fn set_str(ctx: *mut JSContext, obj: JSValue, key: &std::ffi::CStr, val: &str) {
    let c = CString::new(val).unwrap_or_default();
    let v = unsafe { qjs::sofuu_js_new_string(ctx, c.as_ptr()) };
    unsafe { qjs::sofuu_js_set_property_str(ctx, obj, key.as_ptr(), v) };
}

unsafe fn set_bool(ctx: *mut JSContext, obj: JSValue, key: &std::ffi::CStr, val: bool) {
    let v = unsafe { qjs::sofuu_js_new_bool(ctx, if val { 1 } else { 0 }) };
    unsafe { qjs::sofuu_js_set_property_str(ctx, obj, key.as_ptr(), v) };
}

unsafe fn set_num(ctx: *mut JSContext, obj: JSValue, key: &std::ffi::CStr, val: f64) {
    let v = unsafe { qjs::sofuu_js_new_float64(ctx, val) };
    unsafe { qjs::sofuu_js_set_property_str(ctx, obj, key.as_ptr(), v) };
}

fn opts_from_json(j: Option<serde_json::Value>) -> RunOptions {
    let mut opts = RunOptions::default();
    let Some(serde_json::Value::Object(map)) = j else { return opts };
    if let Some(entry) = map.get("entry").and_then(|v| v.as_str()) {
        opts.entry = Some(entry.to_string());
    }
    if let Some(fuel) = map.get("fuel").and_then(|v| v.as_u64()) {
        opts.fuel = fuel;
    }
    if let Some(bytes) = map.get("heapBytes").and_then(|v| v.as_u64()) {
        opts.heap_bytes = bytes as usize;
    }
    if let Some(ticks) = map.get("ticks").and_then(|v| v.as_u64()) {
        opts.ticks = ticks.min(10_000) as u32;
    }
    if let Some(ms) = map.get("tickMs").and_then(|v| v.as_u64()) {
        opts.tick_ms = ms.max(1);
    }
    for key in ["caps", "grants"] {
        match map.get(key) {
            Some(serde_json::Value::Object(o)) => opts.grants.extend(o.keys().cloned()),
            Some(serde_json::Value::Array(a)) => {
                opts.grants.extend(a.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()))
            }
            _ => {}
        }
    }
    if let Some(serde_json::Value::Object(sim)) = map.get("sim") {
        let mut spec = String::new();
        for (k, v) in sim {
            if !spec.is_empty() {
                spec.push(',');
            }
            spec.push_str(k);
            spec.push('=');
            spec.push_str(&match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            });
        }
        let _ = opts.sim.apply_overrides(&spec);
    }
    opts
}

unsafe extern "C" fn js_sfs_run(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    unsafe {
        if argc < 1 {
            let err = KazeError::new("E_KAZE_ARGS", "sfs.run(pathOrSource, args?, opts?) needs a script");
            return error_to_js(ctx, &err);
        }
        let text = js_text(ctx, *argv);
        let args = if argc > 1 {
            match js_json(ctx, *argv.add(1)) {
                Some(serde_json::Value::Object(m)) => m,
                _ => serde_json::Map::new(),
            }
        } else {
            serde_json::Map::new()
        };
        let opts_json = if argc > 2 { js_json(ctx, *argv.add(2)) } else { None };
        let mut opts = opts_from_json(opts_json);
        opts.args = args;

        let result = source_of(&text).and_then(|(src, name)| run_source(&src, &name, &opts));
        match result {
            Ok(report) => report_to_js(ctx, &report),
            Err(err) => error_to_js(ctx, &err),
        }
    }
}

unsafe extern "C" fn js_sfs_check(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    unsafe {
        if argc < 1 {
            let err = KazeError::new("E_KAZE_ARGS", "sfs.check(pathOrSource) needs a script");
            return error_to_js(ctx, &err);
        }
        let text = js_text(ctx, *argv);
        let result = source_of(&text).and_then(|(src, name)| check_source(&src, &name));
        match result {
            Ok(()) => {
                let obj = qjs::sofuu_js_new_object(ctx);
                set_bool(ctx, obj, c"ok", true);
                obj
            }
            Err(err) => error_to_js(ctx, &err),
        }
    }
}

unsafe extern "C" fn js_sfs_compile(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    unsafe {
        if argc < 1 {
            let err = KazeError::new("E_KAZE_ARGS", "sfs.compile(pathOrSource) needs a script");
            return error_to_js(ctx, &err);
        }
        let text = js_text(ctx, *argv);
        let result = source_of(&text).and_then(|(src, name)| prepare(&src, &name));
        match result {
            Ok(prepared) => {
                let bytes = isa::serialize(&prepared.module);
                let obj = qjs::sofuu_js_new_object(ctx);
                set_bool(ctx, obj, c"ok", true);
                set_num(ctx, obj, c"bytes", bytes.len() as f64);
                set_num(ctx, obj, c"items", prepared.module.items.len() as f64);
                set_num(ctx, obj, c"caps", prepared.module.caps.len() as f64);
                set_num(ctx, obj, c"consts", prepared.module.consts.len() as f64);
                obj
            }
            Err(err) => error_to_js(ctx, &err),
        }
    }
}

/// Register `sofuu.sfs`. Called once at engine boot, after the `sofuu`
/// namespace exists.
///
/// # Safety
/// `ctx` must be the live engine context.
pub unsafe fn mod_sfs_register(ctx: *mut JSContext) {
    unsafe {
        let global = qjs::sofuu_js_get_global_object(ctx);
        let mut sofuu_obj = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());
        if qjs::is_undefined(sofuu_obj) {
            sofuu_obj = qjs::sofuu_js_new_object(ctx);
            qjs::sofuu_js_set_property_str(
                ctx,
                global,
                c"sofuu".as_ptr(),
                qjs::sofuu_js_dup_value(ctx, sofuu_obj),
            );
        }
        let ns = qjs::sofuu_js_new_object(ctx);
        let run = qjs::sofuu_js_new_cfunction(ctx, js_sfs_run, c"run".as_ptr(), 3);
        qjs::sofuu_js_set_property_str(ctx, ns, c"run".as_ptr(), run);
        let check = qjs::sofuu_js_new_cfunction(ctx, js_sfs_check, c"check".as_ptr(), 1);
        qjs::sofuu_js_set_property_str(ctx, ns, c"check".as_ptr(), check);
        let compile = qjs::sofuu_js_new_cfunction(ctx, js_sfs_compile, c"compile".as_ptr(), 1);
        qjs::sofuu_js_set_property_str(ctx, ns, c"compile".as_ptr(), compile);
        let version = qjs::sofuu_js_new_string(ctx, c"0.1".as_ptr());
        qjs::sofuu_js_set_property_str(ctx, ns, c"version".as_ptr(), version);
        qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"sfs".as_ptr(), ns);
        qjs::sofuu_js_free_value(ctx, sofuu_obj);
        qjs::sofuu_js_free_value(ctx, global);
    }
}

pub struct KazeSubsystem;

impl crate::domains::Subsystem for KazeSubsystem {
    fn name(&self) -> &'static str {
        "kaze"
    }

    fn register_js(&self, ctx: *mut JSContext) -> Result<(), &'static str> {
        unsafe {
            mod_sfs_register(ctx);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str, opts: &RunOptions) -> RunReport {
        run_source(src, "<test>", opts).unwrap_or_else(|e| panic!("run failed: {}", e.render()))
    }

    fn grants(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_patrol_runs_end_to_end_with_a_deterministic_trace() {
        let src = "needs: drone, log\n\naction patrol(points):\n    log \"Starting patrol\"\n    arm drone\n    takeoff to 15 meters\n    for each point in points:\n        fly to point at 15 meters\n    return to launch\n";
        let mut args = serde_json::Map::new();
        args.insert("points".into(), serde_json::json!([1, 2, 3]));
        let opts = RunOptions { args, grants: grants(&["drone"]), ..Default::default() };
        let report = run(src, &opts);
        assert_eq!(report.entry, "patrol");
        assert_eq!(
            report.actions,
            vec![
                "log.info(\"Starting patrol\") -> nothing",
                "drone.arm -> nothing",
                "drone.takeoff(15) -> nothing",
                "drone.goto(1, 15) -> nothing",
                "drone.goto(2, 15) -> nothing",
                "drone.goto(3, 15) -> nothing",
                "drone.rtl -> nothing",
            ]
        );
        assert_eq!(report.logs.len(), 1);
        assert!(report.fuel_used > 0);
    }

    #[test]
    fn the_same_run_twice_gives_the_same_trace() {
        let src = "needs: rover, lidar\n\ncheck scan():\n    let d = lidar.front_distance()\n    if d < 1 meters:\n        stop rover\n    otherwise:\n        drive forward at 0.5 m/s\n";
        let opts = RunOptions { grants: grants(&["rover", "lidar"]), ..Default::default() };
        let a = run(src, &opts);
        let b = run(src, &opts);
        assert_eq!(a.actions, b.actions);
        assert_eq!(a.fuel_used, b.fuel_used);
    }

    #[test]
    fn an_undeclared_effector_is_refused_before_it_can_move() {
        let src = "needs: drone\naction main():\n    arm drone\n";
        let err = run_source(src, "<test>", &RunOptions::default()).unwrap_err();
        assert_eq!(err.code, "E_CAP_MISSING");
        assert!(err.message.contains("--cap drone=sim"), "{}", err.message);
    }

    #[test]
    fn fuel_stops_a_runaway_loop() {
        let src = "needs: log\naction main():\n    while true:\n        log \"spin\"\n";
        let opts = RunOptions { fuel: 500, ..Default::default() };
        let err = run_source(src, "<test>", &opts).unwrap_err();
        assert_eq!(err.code, "E_TIMEOUT");
    }

    #[test]
    fn the_low_battery_hook_fires_only_below_the_threshold() {
        let src = "needs: drone, log\n\naction main():\n    log \"patrol\"\n\nwhen battery < 20%:\n    log warning \"Battery critical\"\n    return to launch\n";
        let mut sim = SimState::default();
        sim.battery = 15.0;
        let opts = RunOptions { grants: grants(&["drone"]), ticks: 1, sim, ..Default::default() };
        let hot = run(src, &opts);
        assert!(hot.actions.iter().any(|a| a.contains("drone.rtl")), "{:?}", hot.actions);

        let opts2 = RunOptions { grants: grants(&["drone"]), ticks: 1, ..Default::default() };
        let cold = run(src, &opts2);
        assert!(!cold.actions.iter().any(|a| a.contains("drone.rtl")), "{:?}", cold.actions);
    }

    #[test]
    fn every_hundred_ms_fires_on_a_hundred_ms_tick_grid() {
        let src = "needs: sensor, log\n\nevery 100ms:\n    log \"tick\"\n";
        let opts = RunOptions { ticks: 3, tick_ms: 100, ..Default::default() };
        let report = run(src, &opts);
        assert_eq!(report.logs.len(), 3);
        assert_eq!(report.ticks, 3);
    }

    #[test]
    fn a_guardrail_returns_a_verdict_instead_of_a_side_effect() {
        let src = "needs: memory, policy\n\nfilter user_prompt(prompt):\n    if length of prompt > 20:\n        reject with \"too long\"\n    remember prompt as \"last\"\n    allow prompt\n";
        let mut args = serde_json::Map::new();
        args.insert("prompt".into(), serde_json::json!("hello"));
        let report = run(src, &RunOptions { args, ..Default::default() });
        assert_eq!(report.verdict, Some(Verdict::Allow("\"hello\"".to_string())));

        let mut args = serde_json::Map::new();
        args.insert("prompt".into(), serde_json::json!("this prompt is definitely too long"));
        let report = run(src, &RunOptions { args, ..Default::default() });
        assert_eq!(report.verdict, Some(Verdict::Reject("too long".to_string())));
    }

    #[test]
    fn a_policy_deny_from_the_host_stops_the_capability() {
        let src = "needs: drone\naction main():\n    arm drone\n";
        let opts = RunOptions {
            grants: grants(&["drone"]),
            policy: Some(serde_json::json!({ "tools": { "deny": ["kaze.drone.*"] } })),
            ..Default::default()
        };
        let err = run_source(src, "<test>", &opts).unwrap_err();
        assert_eq!(err.code, "E_CAP_DENIED");
    }

    #[test]
    fn out_of_range_arguments_are_rejected_by_the_gate() {
        let src = "needs: drone\naction main():\n    arm drone\n    takeoff to 900 meters\n";
        let opts = RunOptions { grants: grants(&["drone"]), ..Default::default() };
        let err = run_source(src, "<test>", &opts).unwrap_err();
        assert_eq!(err.code, "E_KAZE_RANGE");
    }

    #[test]
    fn a_record_argument_is_read_by_field() {
        // Maps have no literal syntax in v0.1: they arrive as arguments, and
        // `record.field` is how a script reads one.
        let src = "needs: alert\naction main(record):\n    if record.kind == \"fire\":\n        send alert \"evacuate\"\n    return record.level\n";
        let mut args = serde_json::Map::new();
        args.insert("record".into(), serde_json::json!({"kind": "fire", "level": 3}));
        let opts = RunOptions { args, grants: grants(&["alert"]), ..Default::default() };
        let report = run(src, &opts);
        assert_eq!(report.value, "3");
        assert_eq!(report.actions.len(), 1, "{:?}", report.actions);

        // A missing field reads as nothing, not as an error.
        let mut args = serde_json::Map::new();
        args.insert("record".into(), serde_json::json!({"kind": "flood"}));
        let report = run(src, &RunOptions { args, ..Default::default() });
        assert_eq!(report.value, "nothing");
        assert!(report.actions.is_empty());
    }

    #[test]
    fn the_three_argument_goto_is_the_capability_signature() {
        let src = "needs: drone\naction main():\n    arm drone\n    takeoff to 5 meters\n    drone.goto(1, 2, 3)\n";
        let opts = RunOptions { grants: grants(&["drone"]), ..Default::default() };
        let report = run(src, &opts);
        assert!(
            report.actions.iter().any(|a| a == "drone.goto(1, 2, 3) -> nothing"),
            "{:?}",
            report.actions
        );
    }

    #[test]
    fn a_named_pin_reads_its_telemetry() {
        let src = "needs: sensor\naction main():\n    let t = sensor.read(\"temperature\")\n    return t\n";
        let opts = RunOptions { grants: grants(&["sensor"]), ..Default::default() };
        let report = run(src, &opts);
        assert_eq!(report.value, "22");
    }

    #[test]
    fn a_parse_error_names_the_line_and_stage() {
        let err = run_source("needs: log\naction main(): log \"x\"\n", "<t>", &RunOptions::default())
            .unwrap_err();
        assert_eq!(err.code, "E_KAZE_PARSE");
        assert!(err.message.contains("<t>:"), "{}", err.message);
    }

    #[test]
    fn an_ambiguous_entry_is_refused_with_advice() {
        let src = "needs: log\naction a():\n    log \"1\"\naction b():\n    log \"2\"\n";
        let err = run_source(src, "<t>", &RunOptions::default()).unwrap_err();
        assert_eq!(err.code, "E_KAZE_ENTRY");
        assert!(err.message.contains("--entry"), "{}", err.message);
    }

    #[test]
    fn compiled_modules_round_trip_and_execute() {
        let src = "needs: log\naction main():\n    let x = 2 + 3\n    return x * 2\n";
        let prepared = prepare(src, "<t>").unwrap();
        let bytes = isa::serialize(&prepared.module);
        let back = isa::deserialize(&bytes).unwrap();
        let mut sim = SimHost::new(SimState::default());
        let declared = back.needs.clone();
        let mut vm = Vm::new(&back, &mut sim, &declared, &[], None, Limits::default());
        let item = back.item("main").unwrap();
        let out = vm.run_item(item, &[]).unwrap();
        assert_eq!(out, value::Value::Num(10.0));
    }
}
