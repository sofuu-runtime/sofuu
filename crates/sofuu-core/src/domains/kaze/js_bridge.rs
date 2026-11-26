// domains/kaze/js_bridge.rs — QuickJS bindings (`sofuu.sfs` / `sofuu.kaze`)
// and resumable async execution over `sofuu_ffi::kaze::Vm`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::{c_int, c_void};
use std::sync::atomic::{AtomicU64, Ordering};

use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

use super::{
    bind_args_json, c_authorise, c_dispatch, c_item_error, c_poll_deadline, caps, check_source,
    layered_args, prepare, read_source, resolve_backends, resolve_entry, run_source,
    simulated_families, BackendSelection, CShadow, KazeError, Limits, Prepared, RunOptions,
    RunReport, Verdict, VERSION,
};

const ASYNC_INSTRUCTION_SLICE: usize = 256;
const ASYNC_RUN_LIMIT: usize = 64;
const ASYNC_TIMEOUT_DEFAULT_MS: u64 = 120_000;
const ASYNC_TIMEOUT_MAX_MS: u64 = 600_000;
static NEXT_ASYNC_RUN_ID: AtomicU64 = AtomicU64::new(1);

struct AsyncPending {
    id: u64,
    family: String,
    op: String,
    line: u32,
    args: Vec<serde_json::Value>,
}

struct AsyncHook {
    index: usize,
    tag: String,
    period_ms: f64,
}

struct AsyncRun {
    vm: sofuu_ffi::kaze::Vm,
    _shadow: Box<CShadow>,
    source_name: String,
    from_bytecode: bool,
    entry: String,
    entry_line: u32,
    line_map_empty: bool,
    entry_result: Option<String>,
    in_supervisor: bool,
    hooks: Vec<AsyncHook>,
    hook_index: usize,
    tick_index: u32,
    completed_ticks: u32,
    requested_ticks: u32,
    tick_ms: u64,
    context_key: usize,
    simulated_families: Vec<String>,
    backends: Vec<BackendSelection>,
    deadline: std::time::Instant,
    next_req_id: u64,
    pending_req: Option<AsyncPending>,
}

thread_local! {
    static ASYNC_RUNS: RefCell<HashMap<u64, AsyncRun>> = RefCell::new(HashMap::new());
}

enum AsyncStep {
    Yielded { remaining_ms: u64 },
    Capability(AsyncPending, u64),
    Complete(RunReport, Vec<BackendSelection>),
    Failed(KazeError),
}

fn async_advance(run: &mut AsyncRun) -> AsyncStep {
    if std::time::Instant::now() >= run.deadline {
        if let Some(p) = run.pending_req.take() {
            run._shadow.host.note_async_unknown(&p.family, &p.op);
        }
        return AsyncStep::Failed(KazeError::new(
            "E_TIMEOUT",
            "the asynchronous Kaze run deadline elapsed",
        ));
    }
    let remaining_ms = run
        .deadline
        .saturating_duration_since(std::time::Instant::now())
        .as_millis()
        .max(1)
        .min(u64::MAX as u128) as u64;

    let (status, cap_req) = run.vm.step(ASYNC_INSTRUCTION_SLICE);
    match status {
        sofuu_ffi::kaze::SLICE_YIELD => AsyncStep::Yielded { remaining_ms },
        sofuu_ffi::kaze::SLICE_CAP_REQ => {
            if let Err(err) = run._shadow.host.check_async_dispatch() {
                let code = if err.starts_with("E_KAZE_MEMORY") {
                    "E_KAZE_MEMORY"
                } else {
                    "E_KAZE_HOST"
                };
                return AsyncStep::Failed(KazeError::new(code, err));
            }
            let (line, fam, verb, raw_args) = match cap_req {
                Some((f, v, a)) => (run.vm.pending().0, f, v, a),
                None => run.vm.pending(),
            };
            let args: Vec<serde_json::Value> = serde_json::from_str(&raw_args).unwrap_or_default();
            run.next_req_id = run.next_req_id.saturating_add(1);
            let pending = AsyncPending {
                id: run.next_req_id,
                family: fam,
                op: verb,
                line,
                args,
            };
            let step_copy = AsyncPending {
                id: pending.id,
                family: pending.family.clone(),
                op: pending.op.clone(),
                line: pending.line,
                args: pending.args.clone(),
            };
            run.pending_req = Some(pending);
            AsyncStep::Capability(step_copy, remaining_ms)
        }
        sofuu_ffi::kaze::SLICE_DONE => {
            if !run.in_supervisor {
                let val_json: serde_json::Value =
                    serde_json::from_str(&run.vm.value_json()).unwrap_or(serde_json::Value::Null);
                run.entry_result = Some(caps::render_value(&val_json));
                run.in_supervisor = true;
            }
            if let Some((hook_idx, hook_line)) = next_supervisor_hook(run) {
                run.entry_line = hook_line;
                if !run.vm.start(hook_idx, None) {
                    return AsyncStep::Failed(c_item_error(
                        &run.source_name,
                        &run.vm,
                        run.from_bytecode,
                        hook_line,
                        run.line_map_empty,
                    ));
                }
                return AsyncStep::Yielded { remaining_ms };
            }
            let verdict = match run.vm.verdict_kind().as_deref() {
                Some("allow") => Some(Verdict::Allow(run.vm.verdict_text())),
                Some("reject") => Some(Verdict::Reject(run.vm.verdict_text())),
                _ => None,
            };
            AsyncStep::Complete(
                RunReport {
                    entry: run.entry.clone(),
                    value: run
                        .entry_result
                        .clone()
                        .unwrap_or_else(|| "nothing".to_string()),
                    actions: run._shadow.host.actions().iter().map(|a| a.render()).collect(),
                    logs: run._shadow.host.logs().to_vec(),
                    fuel_used: run.vm.fuel_used(),
                    instructions: run.vm.instructions(),
                    ticks: run.completed_ticks,
                    verdict,
                    simulated_families: run.simulated_families.clone(),
                },
                run.backends.clone(),
            )
        }
        sofuu_ffi::kaze::SLICE_ERR => AsyncStep::Failed(c_item_error(
            &run.source_name,
            &run.vm,
            run.from_bytecode,
            run.entry_line,
            run.line_map_empty,
        )),
        _ => AsyncStep::Failed(KazeError::new(
            "E_KAZE_STATE",
            "the Kaze VM returned an unknown slice status",
        )),
    }
}

fn next_supervisor_hook(run: &mut AsyncRun) -> Option<(usize, u32)> {
    while run.tick_index < run.requested_ticks {
        let t_ms = run.tick_index as f64 * run.tick_ms as f64;
        while run.hook_index < run.hooks.len() {
            let h = &run.hooks[run.hook_index];
            run.hook_index += 1;
            let due = match h.tag.as_str() {
                "when" => true,
                "every" => h.period_ms > 0.0 && (t_ms % h.period_ms) == 0.0,
                _ => false,
            };
            if due {
                let line = run.vm.item_line(h.index);
                return Some((h.index, line));
            }
        }
        run.completed_ticks = run.tick_index.saturating_add(1);
        run.tick_index = run.tick_index.saturating_add(1);
        run.hook_index = 0;
    }
    None
}

fn create_async_run(
    ctx: *mut JSContext,
    prepared: Prepared,
    opts: &RunOptions,
) -> Result<(u64, AsyncRun), KazeError> {
    let selected_entry = resolve_entry(&prepared, opts)?;
    let hooks: Vec<AsyncHook> = if opts.ticks > 0 {
        prepared
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| matches!(item.tag.as_str(), "when" | "every"))
            .map(|(index, item)| AsyncHook {
                index,
                tag: item.tag.clone(),
                period_ms: item.period_ms,
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut initial_hook_index = 0usize;
    let in_supervisor = selected_entry.is_none();
    let (start_index, start_line, start_args) = if let Some(entry_name) = &selected_entry {
        let (idx, item) = prepared
            .items
            .iter()
            .enumerate()
            .find(|(_, i)| &i.name == entry_name)
            .ok_or_else(|| {
                KazeError::new(
                    "E_KAZE_ENTRY",
                    format!("there is no '{}' in this script", entry_name),
                )
            })?;
        let merged = layered_args(opts, &item.params)?;
        let args = bind_args_json(&item.params, &merged)?;
        (idx, item.line, Some(args))
    } else {
        if opts.ticks == 0 {
            return Err(KazeError::new(
                "E_KAZE_ASYNC_UNSUPPORTED",
                "asynchronous Kaze run has no entry routine or requested supervisor tick",
            ));
        }
        let mut found = None;
        for (i, h) in hooks.iter().enumerate() {
            let due = match h.tag.as_str() {
                "when" => true,
                "every" => h.period_ms > 0.0,
                _ => false,
            };
            if due {
                found = Some((i, h.index));
                break;
            }
        }
        let Some((pos, idx)) = found else {
            return Err(KazeError::new(
                "E_KAZE_ASYNC_UNSUPPORTED",
                "hook-only program has no supervisor block due on its first tick",
            ));
        };
        initial_hook_index = pos + 1;
        let line = prepared.items.get(idx).map(|i| i.line).unwrap_or(0);
        (idx, line, None)
    };

    let entry = selected_entry.unwrap_or_else(|| "<supervisor>".to_string());
    let active_count = ASYNC_RUNS.with(|runs| runs.borrow().len());
    if active_count >= ASYNC_RUN_LIMIT {
        return Err(KazeError::new(
            "E_KAZE_BUSY",
            "the per-thread asynchronous Kaze run limit has been reached",
        ));
    }

    let declared = prepared.needs.clone();
    let selection = resolve_backends(opts, &prepared.name)?;
    let host = super::build_cap_host(opts, &selection, &declared)?;
    let limits = Limits {
        fuel: opts.fuel.max(1),
        heap_bytes: opts.heap_bytes,
        ..Limits::default()
    };

    let mut shadow = Box::new(CShadow {
        host,
        declared,
        grants: opts.grants.clone(),
        policy: opts.policy.clone(),
        runtime_policy: opts.runtime_policy.clone(),
        async_mode: true,
    });

    let cbs = sofuu_ffi::kaze::HostCbs {
        ctx: (&mut *shadow as *mut CShadow).cast::<c_void>(),
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

    if !vm.start(start_index, start_args.as_deref()) {
        return Err(c_item_error(
            &prepared.name,
            &vm,
            prepared.from_bytecode,
            start_line,
            prepared.line_map_empty,
        ));
    }

    let id = NEXT_ASYNC_RUN_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| KazeError::new("E_KAZE_STATE", "asynchronous run identifiers exhausted"))?;

    if ASYNC_RUNS.with(|runs| runs.borrow().len()) >= ASYNC_RUN_LIMIT {
        return Err(KazeError::new(
            "E_KAZE_BUSY",
            "the per-thread asynchronous Kaze run limit has been reached",
        ));
    }

    Ok((
        id,
        AsyncRun {
            vm,
            _shadow: shadow,
            source_name: prepared.name,
            from_bytecode: prepared.from_bytecode,
            entry,
            entry_line: start_line,
            line_map_empty: prepared.line_map_empty,
            entry_result: None,
            in_supervisor,
            hooks,
            hook_index: initial_hook_index,
            tick_index: 0,
            completed_ticks: 0,
            requested_ticks: opts.ticks,
            tick_ms: opts.tick_ms,
            context_key: ctx as usize,
            simulated_families: simulated_families(&selection),
            backends: selection,
            deadline: std::time::Instant::now() + std::time::Duration::from_millis(opts.timeout_ms),
            next_req_id: 0,
            pending_req: None,
        },
    ))
}

/// Drop any suspended runs before QuickJS tears down its context.
pub fn cancel_async_runs_for_context(ctx: *mut JSContext) {
    let context_key = ctx as usize;
    ASYNC_RUNS.with(|runs| {
        let mut runs = runs.borrow_mut();
        let ids: Vec<u64> = runs
            .iter()
            .filter_map(|(id, run)| (run.context_key == context_key).then_some(*id))
            .collect();
        for id in ids {
            let _ = runs.remove(&id);
        }
    });
}

unsafe fn js_text(ctx: *mut JSContext, v: JSValueConst) -> String {
    let p = unsafe { qjs::sofuu_js_to_cstring(ctx, v) };
    if p.is_null() {
        return String::new();
    }
    let s = unsafe { std::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned();
    unsafe { qjs::sofuu_js_free_cstring(ctx, p) };
    s
}

unsafe fn js_json(ctx: *mut JSContext, v: JSValueConst) -> Option<serde_json::Value> {
    let text = {
        let j = unsafe {
            qjs::JS_JSONStringify(ctx, v, qjs::sofuu_js_undefined(), qjs::sofuu_js_undefined())
        };
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

unsafe fn js_source_of(
    ctx: *mut JSContext,
    input: JSValueConst,
    text: &str,
) -> Result<(String, String), KazeError> {
    if text.contains('\n') {
        return Ok((text.to_string(), "<inline>".to_string()));
    }
    let named_script = text.ends_with(".kaze") || text.ends_with(".kz");
    let path = match crate::rt::fs::fs_path_arg(ctx, input, "kaze", crate::rt::policy::FsKind::Read) {
        Ok(path) => path,
        Err(_) => {
            if !named_script && text.chars().any(char::is_whitespace) {
                return Ok((text.to_string(), "<inline>".to_string()));
            }
            return Err(unsafe { js_fs_denial(ctx) });
        }
    };
    let resolved = path.to_string_lossy();
    if named_script || std::path::Path::new(resolved.as_ref()).exists() {
        let src = read_source(resolved.as_ref())?;
        Ok((src, text.to_string()))
    } else {
        Ok((text.to_string(), "<inline>".to_string()))
    }
}

unsafe fn js_fs_denial(ctx: *mut JSContext) -> KazeError {
    let exception = unsafe { qjs::JS_GetException(ctx) };
    let code_value = unsafe { qjs::sofuu_js_get_property_str(ctx, exception, c"code".as_ptr()) };
    let code_text = unsafe { js_text(ctx, code_value) };
    unsafe { qjs::sofuu_js_free_value(ctx, code_value) };
    let message_value =
        unsafe { qjs::sofuu_js_get_property_str(ctx, exception, c"message".as_ptr()) };
    let message = unsafe { js_text(ctx, message_value) };
    unsafe { qjs::sofuu_js_free_value(ctx, message_value) };
    unsafe { qjs::sofuu_js_free_value(ctx, exception) };
    let code = match code_text.as_str() {
        "E_PATH_OUTSIDE" => "E_PATH_OUTSIDE",
        "E_PERMISSION_DENIED" => "E_PERMISSION_DENIED",
        _ => "E_CAP_DENIED",
    };
    KazeError::new(code, format!("script file access denied: {message}"))
}

unsafe fn js_async_prepared_of(
    ctx: *mut JSContext,
    input: JSValueConst,
    text: &str,
    named: Option<&str>,
) -> Result<Prepared, KazeError> {
    if !text.contains('\n') && text.trim_end().ends_with(".kazebc") {
        let path =
            match unsafe { crate::rt::fs::fs_path_arg(ctx, input, "kaze", crate::rt::policy::FsKind::Read) } {
                Ok(path) => path,
                Err(_) => return Err(unsafe { js_fs_denial(ctx) }),
            };
        let resolved = path.to_string_lossy().into_owned();
        let bytes = std::fs::read(&resolved).map_err(|e| {
            KazeError::new("E_KAZE_IO", format!("cannot read '{}': {}", resolved, e))
        })?;
        let name = named.map(str::to_string).unwrap_or(resolved);
        return super::load_bytecode(bytes, &name);
    }
    let (source, default_name) = unsafe { js_source_of(ctx, input, text) }?;
    let name = named.map(str::to_string).unwrap_or(default_name);
    prepare(&source, &name)
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

unsafe fn json_to_js(ctx: *mut JSContext, value: &serde_json::Value) -> JSValue {
    let Ok(mut text) = serde_json::to_string(value) else {
        return unsafe {
            qjs::JS_ThrowInternalError(ctx, c"Kaze async result could not be serialized".as_ptr())
        };
    };
    let len = text.len();
    text.push('\0');
    unsafe { qjs::JS_ParseJSON(ctx, text.as_ptr().cast(), len, c"<kaze-async>".as_ptr()) }
}

fn report_to_js(ctx: *mut JSContext, report: &RunReport, backends: &[BackendSelection]) -> JSValue {
    unsafe {
        let obj = qjs::sofuu_js_new_object(ctx);
        set_bool(ctx, obj, c"ok", true);
        set_str(ctx, obj, c"entry", &report.entry);
        set_str(ctx, obj, c"result", &report.value);
        set_num(ctx, obj, c"fuel", report.fuel_used as f64);
        set_num(ctx, obj, c"instructions", report.instructions as f64);
        set_num(ctx, obj, c"ticks", report.ticks as f64);
        let simulated = qjs::JS_NewArray(ctx);
        for (i, family) in report.simulated_families.iter().enumerate() {
            let v = qjs::sofuu_js_new_string(
                ctx,
                CString::new(family.as_str()).unwrap_or_default().as_ptr(),
            );
            qjs::JS_SetPropertyUint32(ctx, simulated, i as u32, v);
        }
        qjs::sofuu_js_set_property_str(ctx, obj, c"simulated_families".as_ptr(), simulated);

        let backend_list = qjs::JS_NewArray(ctx);
        for (i, backend) in backends.iter().enumerate() {
            let entry = qjs::sofuu_js_new_object(ctx);
            set_str(ctx, entry, c"family", &backend.family);
            set_str(ctx, entry, c"backend", &backend.backend);
            set_bool(ctx, entry, c"simulated", backend.simulated);
            qjs::JS_SetPropertyUint32(ctx, backend_list, i as u32, entry);
        }
        qjs::sofuu_js_set_property_str(ctx, obj, c"backends".as_ptr(), backend_list);

        let actions = qjs::JS_NewArray(ctx);
        for (i, a) in report.actions.iter().enumerate() {
            let v = qjs::sofuu_js_new_string(
                ctx,
                CString::new(a.as_str()).unwrap_or_default().as_ptr(),
            );
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
        let mut diagnostic = err.diagnostic(None).to_json().to_string();
        let len = diagnostic.len();
        diagnostic.push('\0');
        let value = qjs::JS_ParseJSON(
            ctx,
            diagnostic.as_ptr().cast(),
            len,
            c"<kaze-diagnostic>".as_ptr(),
        );
        if qjs::is_exception(value) {
            qjs::sofuu_js_free_value(ctx, obj);
            return value;
        }
        qjs::sofuu_js_set_property_str(ctx, obj, c"diagnostic".as_ptr(), value);
        obj
    }
}

unsafe fn async_state_only(ctx: *mut JSContext, state: &str) -> JSValue {
    let obj = unsafe { qjs::sofuu_js_new_object(ctx) };
    unsafe { set_str(ctx, obj, c"state", state) };
    obj
}

unsafe fn async_step_value(ctx: *mut JSContext, run_id: Option<u64>, step: AsyncStep) -> JSValue {
    match step {
        AsyncStep::Yielded { remaining_ms } => {
            let obj = unsafe { async_state_only(ctx, "yield") };
            if let Some(id) = run_id {
                unsafe { set_str(ctx, obj, c"runId", &id.to_string()) };
            }
            unsafe { set_num(ctx, obj, c"remainingMs", remaining_ms as f64) };
            obj
        }
        AsyncStep::Capability(request, remaining_ms) => {
            let obj = unsafe { async_state_only(ctx, "capability") };
            if let Some(id) = run_id {
                unsafe { set_str(ctx, obj, c"runId", &id.to_string()) };
            }
            unsafe {
                set_str(ctx, obj, c"requestId", &request.id.to_string());
                set_str(ctx, obj, c"family", &request.family);
                set_str(ctx, obj, c"op", &request.op);
                set_num(ctx, obj, c"line", request.line as f64);
                set_num(ctx, obj, c"remainingMs", remaining_ms as f64);
                let args = json_to_js(ctx, &serde_json::Value::Array(request.args));
                qjs::sofuu_js_set_property_str(ctx, obj, c"args".as_ptr(), args);
            }
            obj
        }
        AsyncStep::Complete(report, backends) => {
            let obj = report_to_js(ctx, &report, &backends);
            unsafe {
                set_str(ctx, obj, c"state", "complete");
                if let Some(id) = run_id {
                    set_str(ctx, obj, c"runId", &id.to_string());
                }
            }
            obj
        }
        AsyncStep::Failed(error) => {
            let obj = error_to_js(ctx, &error);
            if !unsafe { qjs::is_exception(obj) } {
                unsafe {
                    set_str(ctx, obj, c"state", "error");
                    if let Some(id) = run_id {
                        set_str(ctx, obj, c"runId", &id.to_string());
                    }
                }
            }
            obj
        }
    }
}

pub(crate) fn opts_from_json(j: Option<serde_json::Value>) -> Result<RunOptions, KazeError> {
    let mut opts = RunOptions::default();
    opts.timeout_ms = ASYNC_TIMEOUT_DEFAULT_MS;
    let Some(v) = j else {
        return Ok(opts);
    };
    let map = v
        .as_object()
        .ok_or_else(|| KazeError::new("E_KAZE_ARGS", "run options must be an object"))?;
    if let Some(entry) = map.get("entry") {
        let entry = entry
            .as_str()
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| KazeError::new("E_KAZE_ARGS", "entry must be a non-empty string"))?;
        opts.entry = Some(entry.to_string());
    }
    if let Some(ambient) = map.get("ambientArgs") {
        opts.ambient_args = ambient
            .as_bool()
            .ok_or_else(|| KazeError::new("E_KAZE_ARGS", "ambientArgs must be a boolean"))?;
    }
    if let Some(path) = map.get("configPath") {
        let path = path
            .as_str()
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| KazeError::new("E_KAZE_ARGS", "configPath must be a non-empty path"))?;
        opts.config_path = Some(path.to_string());
    }
    if let Some(name) = map.get("sourceName") {
        let name = name
            .as_str()
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| KazeError::new("E_KAZE_ARGS", "sourceName must be a non-empty string"))?;
        opts.source_name = Some(name.to_string());
    }
    if let Some(fuel) = map.get("fuel") {
        opts.fuel = fuel
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or_else(|| KazeError::new("E_KAZE_ARGS", "fuel must be a positive integer"))?;
    }
    if let Some(timeout_ms) = map.get("timeoutMs") {
        let ms = timeout_ms
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or_else(|| KazeError::new("E_KAZE_ARGS", "timeoutMs must be a positive integer"))?;
        if ms > ASYNC_TIMEOUT_MAX_MS {
            return Err(KazeError::new(
                "E_KAZE_ARGS",
                format!("timeoutMs cannot exceed {} ms", ASYNC_TIMEOUT_MAX_MS),
            ));
        }
        opts.timeout_ms = ms;
    }
    if let Some(ticks) = map.get("ticks") {
        let n = ticks
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| KazeError::new("E_KAZE_ARGS", "ticks must be an integer"))?;
        if n > 10_000 {
            return Err(KazeError::new(
                "E_KAZE_ARGS",
                "ticks cannot exceed 10000 in one run",
            ));
        }
        opts.ticks = n;
    }
    if let Some(tick_ms) = map.get("tickMs") {
        opts.tick_ms = tick_ms
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or_else(|| KazeError::new("E_KAZE_ARGS", "tickMs must be a positive integer"))?;
    }
    if let Some(heap_bytes) = map.get("heapBytes") {
        let bytes = heap_bytes
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n > 0)
            .ok_or_else(|| KazeError::new("E_KAZE_ARGS", "heapBytes must be a positive integer"))?;
        opts.heap_bytes = bytes;
    } else if let Some(heap_kb) = map.get("heapKb") {
        let kb = heap_kb
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n > 0)
            .ok_or_else(|| KazeError::new("E_KAZE_ARGS", "heapKb must be a positive integer"))?;
        opts.heap_bytes = kb.saturating_mul(1024);
    }

    let mut add_grant = |opts: &mut RunOptions, family: &str, backend: Option<&str>| -> Result<(), KazeError> {
        if family.trim().is_empty() {
            return Err(KazeError::new("E_CAP_INVALID", "grant family cannot be empty"));
        }
        let backend = backend.unwrap_or("").trim();
        if let Err(d) = crate::rt::hardware::validate_grant(family, backend) {
            return Err(KazeError::new(
                d.code,
                format!("{} ({})", d.reason, d.rule),
            ));
        }
        let resolved = caps::family_of(family);
        if !opts.grants.iter().any(|g| g == family) {
            opts.grants.push(family.to_string());
        }
        if family != resolved && !opts.grants.iter().any(|g| g == resolved) {
            opts.grants.push(resolved.to_string());
        }
        match opts.backend.iter().position(|(f, _)| caps::family_of(f) == resolved) {
            Some(index) => {
                let current = opts.backend[index].1.clone();
                if !current.is_empty() && !backend.is_empty() && current != backend {
                    return Err(KazeError::new(
                        "E_CAP_INVALID",
                        format!(
                            "'{}' was granted two different backends ('{}' and '{}')",
                            resolved, current, backend
                        ),
                    ));
                }
                if current.is_empty() && !backend.is_empty() {
                    opts.backend[index].1 = backend.to_string();
                }
            }
            None => opts.backend.push((family.to_string(), backend.to_string())),
        }
        Ok(())
    };

    for key in ["caps", "grants"] {
        match map.get(key) {
            Some(serde_json::Value::Object(o)) => {
                for (family, backend) in o {
                    match backend {
                        serde_json::Value::String(value) if !value.trim().is_empty() => {
                            add_grant(&mut opts, family, Some(value))?
                        }
                        serde_json::Value::Bool(true) | serde_json::Value::Null => {
                            add_grant(&mut opts, family, None)?
                        }
                        _ => {
                            return Err(KazeError::new(
                                "E_CAP_INVALID",
                                format!("{} backend for '{}' must be text or true", key, family),
                            ))
                        }
                    }
                }
            }
            Some(serde_json::Value::Array(a)) => {
                for grant in a {
                    let grant = grant.as_str().ok_or_else(|| {
                        KazeError::new("E_CAP_INVALID", format!("{} entries must be text", key))
                    })?;
                    add_grant(&mut opts, grant, None)?;
                }
            }
            None => {}
            _ => {
                return Err(KazeError::new(
                    "E_CAP_INVALID",
                    format!("{} must be an array or object", key),
                ))
            }
        }
    }

    if map.contains_key("policy") && map.contains_key("policyJson") {
        return Err(KazeError::new(
            "E_CAP_DENIED",
            "specify policy or policyJson, not both",
        ));
    }
    let parsed_policy = if let Some(raw) = map.get("policyJson") {
        let raw = raw
            .as_str()
            .ok_or_else(|| KazeError::new("E_CAP_DENIED", "policyJson must be text"))?;
        Some(
            crate::rt::policy::parse_policy_strict(raw)
                .map_err(|e| KazeError::new("E_CAP_DENIED", format!("invalid policy JSON: {}", e)))?,
        )
    } else {
        map.get("policy").cloned()
    };
    if let Some(policy) = parsed_policy {
        crate::rt::policy::validate_policy(&policy)
            .map_err(|e| KazeError::new("E_CAP_DENIED", format!("invalid policy: {}", e)))?;
        opts.policy = Some(policy);
    }
    if let Some(pins) = map.get("pins") {
        let pins = pins
            .as_object()
            .ok_or_else(|| KazeError::new("E_KAZE_ARGS", "pins must be an object"))?;
        for (name, value) in pins {
            let line = value
                .as_u64()
                .and_then(|n| u16::try_from(n).ok())
                .ok_or_else(|| {
                    KazeError::new(
                        "E_KAZE_ARGS",
                        format!("pin '{}' must be an unsigned 16-bit integer", name),
                    )
                })?;
            opts.pins.push((name.clone(), line));
        }
    }
    if let Some(value) = map.get("sim") {
        let sim = value
            .as_object()
            .ok_or_else(|| KazeError::new("E_KAZE_ARGS", "sim must be an object"))?;
        if !opts
            .backend
            .iter()
            .any(|(family, backend)| backend == "sim" && matches!(caps::family_of(family), "drone" | "rover"))
        {
            return Err(KazeError::new(
                "E_CAP_INVALID",
                "sim requires an explicit drone=sim or rover=sim testing backend",
            ));
        }
        let mut spec = String::new();
        for (k, v) in sim {
            let value = match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            if k.contains(',') || k.contains('=') || value.contains(',') {
                return Err(KazeError::new(
                    "E_CAP_INVALID",
                    format!("sim override '{}' cannot contain a comma or an equals sign", k),
                ));
            }
            if !spec.is_empty() {
                spec.push(',');
            }
            spec.push_str(k);
            spec.push('=');
            spec.push_str(&value);
        }
        opts.sim.apply_overrides(&spec).map_err(|e| {
            let detail = e.strip_prefix("E_CAP_INVALID: ").unwrap_or(&e);
            KazeError::new("E_CAP_INVALID", detail)
        })?;
    }
    Ok(opts)
}

unsafe extern "C" fn js_sfs_run(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    unsafe {
        if argc < 1 {
            return error_to_js(
                ctx,
                &KazeError::new("E_KAZE_ARGS", "sfs.run(pathOrSource, args?, opts?) needs a script"),
            );
        }
        let text = js_text(ctx, *argv);
        let args = if argc > 1 && !qjs::is_undefined(*argv.add(1)) && !qjs::is_null(*argv.add(1)) {
            match js_json(ctx, *argv.add(1)) {
                Some(serde_json::Value::Object(m)) => m,
                _ => {
                    return error_to_js(
                        ctx,
                        &KazeError::new("E_KAZE_ARGS", "routine arguments must be a JSON-compatible object"),
                    )
                }
            }
        } else {
            serde_json::Map::new()
        };
        let opts_json = if argc > 2 && !qjs::is_undefined(*argv.add(2)) && !qjs::is_null(*argv.add(2)) {
            match js_json(ctx, *argv.add(2)) {
                Some(value) => Some(value),
                None => {
                    return error_to_js(
                        ctx,
                        &KazeError::new("E_KAZE_ARGS", "run options must be valid JSON-compatible values"),
                    )
                }
            }
        } else {
            None
        };
        let mut opts = match opts_from_json(opts_json) {
            Ok(opts) => opts,
            Err(err) => return error_to_js(ctx, &err),
        };
        opts.args = args;
        opts.runtime_policy = Some(crate::rt::policy::policy_json(ctx));
        let (src, name) = match js_source_of(ctx, *argv, &text) {
            Ok(value) => value,
            Err(err) => return error_to_js(ctx, &err),
        };
        let backends = match resolve_backends(&opts, &name) {
            Ok(selection) => selection,
            Err(err) => return error_to_js(ctx, &err),
        };
        match run_source(&src, &name, &opts) {
            Ok(report) => report_to_js(ctx, &report, &backends),
            Err(err) => error_to_js(ctx, &err),
        }
    }
}

unsafe fn js_async_prepare(
    ctx: *mut JSContext,
    argc: c_int,
    argv: *const JSValueConst,
) -> Result<(Prepared, RunOptions), KazeError> {
    if argc < 1 {
        return Err(KazeError::new(
            "E_KAZE_ARGS",
            "sfs.runAsync(pathOrSource, args?, opts?) needs a script",
        ));
    }
    let input = unsafe { js_text(ctx, *argv) };
    let args = if argc > 1 && !unsafe { qjs::is_undefined(*argv.add(1)) || qjs::is_null(*argv.add(1)) } {
        match unsafe { js_json(ctx, *argv.add(1)) } {
            Some(serde_json::Value::Object(map)) => map,
            _ => {
                return Err(KazeError::new(
                    "E_KAZE_ARGS",
                    "routine arguments must be a JSON-compatible object",
                ))
            }
        }
    } else {
        serde_json::Map::new()
    };
    let opts_json =
        if argc > 2 && !unsafe { qjs::is_undefined(*argv.add(2)) || qjs::is_null(*argv.add(2)) } {
            Some(unsafe { js_json(ctx, *argv.add(2)) }.ok_or_else(|| {
                KazeError::new("E_KAZE_ARGS", "run options must be valid JSON-compatible values")
            })?)
        } else {
            None
        };
    let mut opts = opts_from_json(opts_json)?;
    opts.args = args;
    opts.runtime_policy = Some(crate::rt::policy::policy_json(ctx));
    let prepared = unsafe { js_async_prepared_of(ctx, *argv, &input, opts.source_name.as_deref()) }?;
    Ok((prepared, opts))
}

unsafe extern "C" fn js_sfs_async_start(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let (prepared, opts) = match unsafe { js_async_prepare(ctx, argc, argv) } {
        Ok(value) => value,
        Err(error) => return unsafe { async_step_value(ctx, None, AsyncStep::Failed(error)) },
    };
    let (run_id, mut run) = match create_async_run(ctx, prepared, &opts) {
        Ok(value) => value,
        Err(error) => return unsafe { async_step_value(ctx, None, AsyncStep::Failed(error)) },
    };
    let step = async_advance(&mut run);
    let pending = matches!(&step, AsyncStep::Yielded { .. } | AsyncStep::Capability(_, _));
    if pending {
        ASYNC_RUNS.with(|runs| {
            runs.borrow_mut().insert(run_id, run);
        });
    }
    unsafe { async_step_value(ctx, Some(run_id), step) }
}

unsafe extern "C" fn js_sfs_async_resume(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return unsafe {
            async_step_value(
                ctx,
                None,
                AsyncStep::Failed(KazeError::new("E_KAZE_ARGS", "invalid async continuation response")),
            )
        };
    }
    let Some(run_id) = (unsafe { js_text(ctx, *argv) }).parse::<u64>().ok() else {
        return unsafe {
            async_step_value(
                ctx,
                None,
                AsyncStep::Failed(KazeError::new("E_KAZE_ARGS", "invalid async run identifier")),
            )
        };
    };
    let Some(request_id) = (unsafe { js_text(ctx, *argv.add(1)) }).parse::<u64>().ok() else {
        return unsafe {
            async_step_value(
                ctx,
                Some(run_id),
                AsyncStep::Failed(KazeError::new("E_KAZE_ARGS", "invalid async request identifier")),
            )
        };
    };
    let mut run = match ASYNC_RUNS.with(|runs| runs.borrow_mut().remove(&run_id)) {
        Some(run) => run,
        None => return unsafe { async_state_only(ctx, "late") },
    };
    if run.context_key != ctx as usize {
        ASYNC_RUNS.with(|runs| {
            runs.borrow_mut().insert(run_id, run);
        });
        return unsafe {
            async_step_value(
                ctx,
                Some(run_id),
                AsyncStep::Failed(KazeError::new(
                    "E_KAZE_STATE",
                    "async continuation belongs to a different runtime context",
                )),
            )
        };
    }
    if std::time::Instant::now() >= run.deadline {
        if let Some(p) = run.pending_req.take() {
            run._shadow.host.note_async_unknown(&p.family, &p.op);
        }
        return unsafe {
            async_step_value(
                ctx,
                Some(run_id),
                AsyncStep::Failed(KazeError::at(
                    "E_TIMEOUT",
                    "the asynchronous Kaze run deadline elapsed; a dispatched effect may have an unknown outcome",
                    &run.source_name,
                    0,
                    None,
                )),
            )
        };
    }

    if request_id == 0 {
        if run.pending_req.is_some() {
            ASYNC_RUNS.with(|runs| {
                runs.borrow_mut().insert(run_id, run);
            });
            return unsafe {
                async_step_value(
                    ctx,
                    Some(run_id),
                    AsyncStep::Failed(KazeError::new(
                        "E_KAZE_STATE",
                        "cannot advance while a capability response is pending",
                    )),
                )
            };
        }
        let step = async_advance(&mut run);
        let pending = matches!(&step, AsyncStep::Yielded { .. } | AsyncStep::Capability(_, _));
        if pending {
            ASYNC_RUNS.with(|runs| {
                runs.borrow_mut().insert(run_id, run);
            });
        }
        return unsafe { async_step_value(ctx, Some(run_id), step) };
    }

    let Some(active) = run.pending_req.take() else {
        let name = run.source_name.clone();
        return unsafe {
            async_step_value(
                ctx,
                Some(run_id),
                AsyncStep::Failed(KazeError::at(
                    "E_KAZE_STATE",
                    "unexpected capability continuation response",
                    &name,
                    0,
                    None,
                )),
            )
        };
    };
    if active.id != request_id {
        run.pending_req = Some(active);
        ASYNC_RUNS.with(|runs| {
            runs.borrow_mut().insert(run_id, run);
        });
        return unsafe {
            async_step_value(
                ctx,
                Some(run_id),
                AsyncStep::Failed(KazeError::new(
                    "E_KAZE_STATE",
                    "asynchronous capability request identifier mismatch",
                )),
            )
        };
    }

    let err_arg = argc > 3 && !unsafe { qjs::is_undefined(*argv.add(3)) || qjs::is_null(*argv.add(3)) };
    if err_arg {
        let error = unsafe { js_json(ctx, *argv.add(3)) }.unwrap_or(serde_json::Value::Null);
        let raw_code = error
            .get("code")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("E_KAZE_HOST");
        let code = async_backend_error_code(raw_code);
        let message = error
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("asynchronous capability failed");
        run._shadow
            .host
            .note_async_completion(&active.family, &active.op, false);
        let formatted = format!("{code}: {message}");
        let _ = run.vm.supply(None, Some(&formatted));
    } else {
        let value = if argc > 2 && !unsafe { qjs::is_undefined(*argv.add(2)) } {
            unsafe { js_json(ctx, *argv.add(2)) }
        } else {
            Some(serde_json::Value::Null)
        };
        match value {
            Some(val) => {
                run._shadow
                    .host
                    .note_async_completion(&active.family, &active.op, true);
                let json_str = serde_json::to_string(&val).unwrap_or_else(|_| "null".to_string());
                let _ = run.vm.supply(Some(&json_str), None);
            }
            None => {
                run._shadow
                    .host
                    .note_async_completion(&active.family, &active.op, false);
                let _ = run.vm.supply(
                    None,
                    Some("E_KAZE_HOST: asynchronous capability returned a non-JSON result"),
                );
            }
        }
    }

    let step = async_advance(&mut run);
    let pending = matches!(&step, AsyncStep::Yielded { .. } | AsyncStep::Capability(_, _));
    if pending {
        ASYNC_RUNS.with(|runs| {
            runs.borrow_mut().insert(run_id, run);
        });
    }
    unsafe { async_step_value(ctx, Some(run_id), step) }
}

fn async_backend_error_code(code: &str) -> &'static str {
    match code {
        "E_CAP_MISSING" | "E_TOOL" | "E_MCP_CONFIG" | "E_MCP_SECRET_UNAVAILABLE"
        | "E_MCP_SERVER_NOT_FOUND" | "E_MCP_TOOL_NOT_FOUND" | "E_MCP_AMBIGUOUS"
        | "E_MCP_METHOD_NOT_SUPPORTED" | "E_MCP_FETCH_UNAVAILABLE"
        | "E_MCP_HTTP_SERVER_UNAVAILABLE" => "E_CAP_MISSING",
        "E_CAP_DENIED" | "E_POLICY_DENIED" | "E_POLICY_APPROVAL_REQUIRED"
        | "E_POLICY_UNAVAILABLE" | "E_PERMISSION_DENIED" | "E_PATH_OUTSIDE" => "E_CAP_DENIED",
        "E_KAZE_ARGS" | "E_SCHEMA" => "E_KAZE_ARGS",
        "E_KAZE_RESULT" => "E_KAZE_RESULT",
        "E_KAZE_MEMORY" => "E_KAZE_MEMORY",
        "E_KAZE_OUTCOME_UNKNOWN" | "E_MCP_OUTCOME_UNKNOWN" => "E_KAZE_OUTCOME_UNKNOWN",
        "E_TIMEOUT" | "E_MCP_TIMEOUT" => "E_TIMEOUT",
        _ => "E_KAZE_HOST",
    }
}

unsafe extern "C" fn js_sfs_async_cancel(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return unsafe { qjs::sofuu_js_new_bool(ctx, 0) };
    }
    let Some(run_id) = (unsafe { js_text(ctx, *argv) }).parse::<u64>().ok() else {
        return unsafe { qjs::sofuu_js_new_bool(ctx, 0) };
    };
    let Some(mut run) = ASYNC_RUNS.with(|runs| runs.borrow_mut().remove(&run_id)) else {
        return unsafe { qjs::sofuu_js_new_bool(ctx, 0) };
    };
    if run.context_key != ctx as usize {
        ASYNC_RUNS.with(|runs| {
            runs.borrow_mut().insert(run_id, run);
        });
        return unsafe { qjs::sofuu_js_new_bool(ctx, 0) };
    }
    if let Some(p) = run.pending_req.take() {
        run._shadow.host.note_async_unknown(&p.family, &p.op);
    }
    unsafe { qjs::sofuu_js_new_bool(ctx, 1) }
}

unsafe extern "C" fn js_sfs_check(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    unsafe {
        if argc < 1 {
            return error_to_js(
                ctx,
                &KazeError::new("E_KAZE_ARGS", "sfs.check(pathOrSource) needs a script"),
            );
        }
        let text = js_text(ctx, *argv);
        match js_source_of(ctx, *argv, &text).and_then(|(src, name)| check_source(&src, &name)) {
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
            return error_to_js(
                ctx,
                &KazeError::new("E_KAZE_ARGS", "sfs.compile(pathOrSource) needs a script"),
            );
        }
        let text = js_text(ctx, *argv);
        match js_source_of(ctx, *argv, &text).and_then(|(src, name)| prepare(&src, &name)) {
            Ok(prepared) => {
                let obj = qjs::sofuu_js_new_object(ctx);
                set_bool(ctx, obj, c"ok", true);
                set_num(ctx, obj, c"bytes", prepared.bytecode.len() as f64);
                set_num(ctx, obj, c"items", prepared.items.len() as f64);
                set_num(ctx, obj, c"caps", prepared.cap_count as f64);
                set_num(ctx, obj, c"consts", prepared.const_count as f64);
                obj
            }
            Err(err) => error_to_js(ctx, &err),
        }
    }
}

/// Register `sofuu.sfs` and `sofuu.kaze` once at engine boot.
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
        let async_start =
            qjs::sofuu_js_new_cfunction(ctx, js_sfs_async_start, c"_asyncStart".as_ptr(), 3);
        qjs::sofuu_js_set_property_str(ctx, ns, c"_asyncStart".as_ptr(), async_start);
        let async_resume =
            qjs::sofuu_js_new_cfunction(ctx, js_sfs_async_resume, c"_asyncResume".as_ptr(), 4);
        qjs::sofuu_js_set_property_str(ctx, ns, c"_asyncResume".as_ptr(), async_resume);
        let async_cancel =
            qjs::sofuu_js_new_cfunction(ctx, js_sfs_async_cancel, c"_asyncCancel".as_ptr(), 1);
        qjs::sofuu_js_set_property_str(ctx, ns, c"_asyncCancel".as_ptr(), async_cancel);
        let check = qjs::sofuu_js_new_cfunction(ctx, js_sfs_check, c"check".as_ptr(), 1);
        qjs::sofuu_js_set_property_str(ctx, ns, c"check".as_ptr(), check);
        let compile = qjs::sofuu_js_new_cfunction(ctx, js_sfs_compile, c"compile".as_ptr(), 1);
        qjs::sofuu_js_set_property_str(ctx, ns, c"compile".as_ptr(), compile);

        let version_c = CString::new(VERSION).unwrap_or_default();
        let version = qjs::sofuu_js_new_string(ctx, version_c.as_ptr());
        qjs::sofuu_js_set_property_str(ctx, ns, c"version".as_ptr(), version);
        let gate = qjs::sofuu_js_new_string(ctx, c"fail-closed-1".as_ptr());
        qjs::sofuu_js_set_property_str(ctx, ns, c"capabilityGate".as_ptr(), gate);
        let boundary = qjs::sofuu_js_new_string(ctx, c"fs-and-args-1".as_ptr());
        qjs::sofuu_js_set_property_str(ctx, ns, c"embeddedBoundary".as_ptr(), boundary);

        qjs::sofuu_js_set_property_str(
            ctx,
            sofuu_obj,
            c"sfs".as_ptr(),
            qjs::sofuu_js_dup_value(ctx, ns),
        );
        qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"kaze".as_ptr(), ns);
        qjs::sofuu_js_free_value(ctx, sofuu_obj);
        qjs::sofuu_js_free_value(ctx, global);

        let source = crate::shipped::kaze_async_js();
        let driver = CString::new(source).unwrap_or_default();
        let evaluated = qjs::JS_Eval(
            ctx,
            driver.as_ptr(),
            driver.as_bytes().len(),
            c"<kaze-async-driver>".as_ptr(),
            qjs::JS_EVAL_TYPE_GLOBAL,
        );
        if qjs::is_exception(evaluated) {
            qjs::js_std_dump_error(ctx);
        }
        qjs::sofuu_js_free_value(ctx, evaluated);
    }
}
