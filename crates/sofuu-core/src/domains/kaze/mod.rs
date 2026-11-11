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

#[cfg(test)]
mod fail_closed_tests {
    use super::*;
    use serde_json::json;

    fn allowed(family: &str) -> RunOptions {
        RunOptions {
            grants: vec![family.into()],
            policy: Some(json!({"tools": {"allow": ["kaze.*"]}})),
            ..Default::default()
        }
    }

    #[test]
    fn pure_program_needs_no_grant_policy_or_backend() {
        let report = run_source("action main():\n    return 2 + 3\n", "<pure>", &RunOptions::default()).unwrap();
        assert_eq!(report.value, "5");
    }

    #[test]
    fn missing_backends_fail_for_source_and_bytecode() {
        for (family, expression) in [
            ("ai", "ai.prompt(\"hello\")"),
            ("ai", "ai.complete(\"hello\")"),
            ("ai", "ai.classify(\"hello\", [\"hello\"])"),
            ("tool", "tool.call(\"lookup\")"),
            ("tool", "tool.run(\"lookup\")"),
            ("memory", "memory.save(\"key\", \"value\")"),
            ("memory", "memory.recall(\"key\")"),
            ("alert", "alert.send(\"message\")"),
            ("sensor", "sensor.temperature()"),
            ("device", "device.read_gpio(1)"),
        ] {
            let source = format!("needs: {family}\naction main():\n    return {expression}\n");
            let opts = allowed(family);
            let error = run_source(&source, "<source>", &opts).unwrap_err();
            assert_eq!(error.code, "E_CAP_MISSING", "{expression}: {}", error.message);
            assert!(error.message.contains(family));
            let prepared = prepare(&source, "<source>").unwrap();
            let module = isa::deserialize(&isa::serialize(&prepared.module)).unwrap();
            let error = run_module(module, "<bytecode>", &opts).unwrap_err();
            assert_eq!(error.code, "E_CAP_MISSING", "{expression}: {}", error.message);
        }
    }

    #[test]
    fn grants_and_valid_policy_are_required_before_dispatch() {
        let source = "needs: ai\naction main():\n    return ai.prompt(\"hello\")\n";
        let missing_grant = RunOptions { policy: Some(json!({"tools": {"deny": []}})), ..Default::default() };
        assert_eq!(run_source(source, "<t>", &missing_grant).unwrap_err().code, "E_CAP_MISSING");
        for policy in [None, Some(json!({})), Some(json!({"tools": {}})), Some(json!({"tools": {"allow": []}})), Some(json!({"fs": {"write": "off"}})), Some(json!({"tools": {"denny": []}})), Some(json!({"tools": {"deny": ["kaze.ai.*"]}}))] {
            let opts = RunOptions { grants: vec!["ai".into()], policy, ..Default::default() };
            assert_eq!(run_source(source, "<t>", &opts).unwrap_err().code, "E_CAP_DENIED");
        }
    }

    #[test]
    fn simulation_requires_selection_and_is_labelled() {
        let source = "needs: drone\naction main():\n    arm drone\n    return 1\n";
        let mut opts = allowed("drone");
        assert_eq!(run_source(source, "<t>", &opts).unwrap_err().code, "E_CAP_MISSING");
        opts.backend.push(("drone".into(), "sim".into()));
        let report = run_source(source, "<t>", &opts).unwrap();
        assert_eq!(report.simulated_families, vec!["drone"]);
        assert!(report.render_plain().contains("simulation (testing): drone"));
        assert_eq!(report.actions.len(), 1);
        opts.backend[0].1 = "uart:/nonexistent".into();
        assert_eq!(run_source(source, "<t>", &opts).unwrap_err().code, "E_CAP_DENIED");
    }

    #[test]
    fn json_options_preserve_policy_grants_and_reject_invalid_overrides() {
        let opts = opts_from_json(Some(json!({
            "caps": {"drone": "sim"}, "grants": ["drone"],
            "policy": {"tools": {"deny": []}}, "sim": {"battery": 42}
        }))).unwrap();
        assert_eq!(opts.backend, vec![("drone".into(), "sim".into())]);
        assert_eq!(opts.sim.battery, 42.0);
        for invalid in [
            json!({"grants": [42]}),
            json!({"caps": {"ai": "sim"}}),
            json!({"sim": {"battery": 42}}),
            json!({"caps": {"drone": "sim"}, "sim": {"ai": "answer"}}),
            json!({"policy": {"tools": {"denny": []}}}),
            json!({"policyJson": "{\"tools\":{\"deny\":[],\"deny\":[]}}"}),
        ] {
            assert!(opts_from_json(Some(invalid)).is_err());
        }
    }

    #[test]
    fn argument_conversion_refuses_text_that_exceeds_the_arena() {
        let mut arena = arena::Arena::new(4);
        let mut heap = value::Heap::new(1024);
        assert!(json_to_value(&json!("too long"), &mut arena, &mut heap).is_err());

        let mut arena = arena::Arena::new(4);
        let mut heap = value::Heap::new(1024);
        assert!(json_to_value(&json!({"too long": true}), &mut arena, &mut heap).is_err());

        let mut arena = arena::Arena::new(4);
        let mut heap = value::Heap::new(1024);
        assert!(json_to_value(&json!(["ok", "long"]), &mut arena, &mut heap).is_err());
    }
}

#[derive(Debug, Clone)]
pub struct KazeError {
    pub code: String,
    pub message: String,
    /// Verified source location when the parser, checker, compiler or VM had
    /// one. `None` when the failure is not tied to a source line.
    pub location: Option<KazeLocation>,
}

/// A Kaze source location. `column` is present only where the front end
/// actually tracked one (the lexer/parser); a checker/compiler/VM error keeps
/// its line and honestly reports the column as unavailable.
#[derive(Debug, Clone)]
pub struct KazeLocation {
    pub source: String,
    pub line: u32,
    pub column: Option<u32>,
    /// The stored line is the routine start, not the failing instruction.
    pub routine_start: bool,
    /// The source line was loaded from a bytecode artifact, not its source.
    pub bytecode_source: bool,
}

impl KazeError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        KazeError { code: code.to_string(), message: message.into(), location: None }
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
            location: Some(KazeLocation { source: source.into(), line, column, routine_start: false, bytecode_source: false }),
        }
    }

    pub fn render(&self) -> String {
        format!("{}: {}", self.code, self.message)
    }

    /// The shared diagnostic view. Source excerpts are not emitted here.
    pub fn diagnostic(
        &self,
        _source_text: Option<&str>,
    ) -> crate::rt::diag_contract::Diagnostic {
        let mut d =
            crate::rt::diag_contract::Diagnostic::from_code(&self.code, self.detail());
        if let Some(loc) = self.location.as_ref() {
            let mut location = crate::rt::diag_contract::SourceLocation::at(
                loc.source.clone(),
                loc.line,
                loc.column,
            );
            location.label = match (loc.routine_start, loc.bytecode_source) {
                (true, true) => Some("bytecode routine start; instruction line and original source unavailable".into()),
                (true, false) => Some("routine start; instruction line unavailable".into()),
                (false, true) => Some("source line stored in bytecode; original source unavailable".into()),
                (false, false) => None,
            };
            d = d.with_location(location);
        }
        /* A checker failure can carry several errors, newline-joined in the
         * legacy `message`. The versioned message is the first one; that is
         * single-line and cannot embed a second location. The legacy field
         * keeps every error for callers that print it. */
        d
    }

    /// The message without the legacy `source:line:col: ` prefix, so the
    /// diagnostic does not print the location twice. The legacy `message`
    /// field is untouched for callers that already print it.
    fn detail(&self) -> String {
        /* Only the checker aggregates several errors into one message. Its
         * first error is the versioned message; every other Kaze failure
         * keeps its full, possibly multi-line, guidance. */
        let first = if self.code == "E_KAZE_CHECK" {
            self.message.lines().next().unwrap_or("")
        } else {
            self.message.as_str()
        };
        if let Some(loc) = self.location.as_ref() {
            let with_col = format!(
                "{}:{}:{}: ",
                loc.source,
                loc.line,
                loc.column.unwrap_or(0)
            );
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

/// Remove the legacy ` (line N)` suffix once the location is carried as its
/// own field. Only the exact line number is removed; a message that happens
/// to mention another line stays untouched.
fn strip_line_suffix(msg: &str, line: u32) -> String {
    if line == 0 {
        return msg.to_string();
    }
    let suffix = format!(" (line {line})");
    msg.strip_suffix(&suffix).unwrap_or(msg).to_string()
}

/// A compiled script plus the source name it came from.
pub struct Prepared {
    pub module: Module,
    pub name: String,
    pub from_bytecode: bool,
}

/// Read a script from disk.
///
/// KZ-041 (AUDIT-2026-11-08): a UTF-8 BOM is not Kaze syntax, but every editor
/// on Windows writes one, and the parser then reported a parse error on line 1
/// column 1 of an otherwise valid script. The BOM is dropped here, and CRLF is
/// normalised so line numbers match what the operator sees.
pub fn read_source(path: &str) -> Result<String, KazeError> {
    let bytes = std::fs::read(path)
        .map_err(|e| KazeError::new("E_KAZE_IO", format!("cannot read '{}': {}", path, e)))?;
    decode_source(bytes, path)
}

/// Decode an already-read source, including named-pipe input, exactly once.
pub fn decode_source(bytes: Vec<u8>, path: &str) -> Result<String, KazeError> {
    /* KZ-092 (AUDIT-2026-11-08): invalid UTF-8 used to surface as an E_KAZE_IO
     * "stream did not contain valid UTF-8" — i.e. an I/O failure for what is a
     * parse problem in the file itself. Report the byte offset as a parse
     * error so the operator knows where to look. */
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

/// Parse, check and compile. This is the whole front end in one call.
pub fn prepare(src: &str, name: &str) -> Result<Prepared, KazeError> {
    let program = parser::parse(src).map_err(|e| {
        KazeError::at(e.code, e.render(name), name, e.line, Some(e.col))
    })?;
    checker::check(&program).map_err(|errors| {
        let mut message = String::new();
        for (i, err) in errors.iter().enumerate() {
            if i > 0 {
                message.push_str("\n");
            }
            message.push_str(&err.render(name));
        }
        let first_line = errors.first().map(|e| e.line).unwrap_or(0);
        KazeError::at("E_KAZE_CHECK", message, name, first_line, None)
    })?;
    let module = compiler::compile(&program)
        .map_err(|e| KazeError::at("E_KAZE_COMPILE", e.render(name), name, e.line, None))?;
    Ok(Prepared { module, name: name.to_string(), from_bytecode: false })
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
    /// CLI-owned configuration and environment argument layers. Embedded JS
    /// callers provide their arguments explicitly and must not inherit them.
    pub ambient_args: bool,
    /// CLI-selected parameter file; `None` uses the optional default file.
    pub config_path: Option<String>,
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
    /// R7 policy layered over the capability gate. `None` denies capability calls.
    pub policy: Option<serde_json::Value>,
    /// Additional host policy, evaluated as a separate restriction for each
    /// capability; an embedded caller cannot broaden it with `policy`.
    pub runtime_policy: Option<serde_json::Value>,
    /// `device name -> GPIO line` (`--pin fan=17`). Non-empty puts the real
    /// pins behind the `device` family (Kaze v0.2, P-K1).
    pub pins: Vec<(String, u16)>,
    /// `family -> device`. A device other than `sim` (for example
    /// `drone -> udp:127.0.0.1:14550`) puts the live backend behind that family
    /// (Kaze v0.2, P-K1). An empty backend does not select a simulator.
    pub backend: Vec<(String, String)>,
}

impl Default for RunOptions {
    fn default() -> Self {
        RunOptions {
            entry: None,
            args: serde_json::Map::new(),
            ambient_args: true,
            config_path: None,
            grants: Vec::new(),
            fuel: Limits::default().fuel,
            heap_bytes: value::Heap::DEFAULT_LIMIT,
            ticks: 0,
            tick_ms: 100,
            sim: SimState::default(),
            policy: None,
            runtime_policy: None,
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
    /// Explicitly selected physical testing simulators, not real hardware.
    pub simulated_families: Vec<String>,
}

impl RunReport {
    /// The canonical text form of a run — the line a mission review reads and
    /// the exact bytes the golden corpus stores. The CLI adds colour; nothing
    /// else about the shape changes.
    pub fn render_plain(&self) -> String {
        let mut out = String::new();
        if !self.simulated_families.is_empty() {
            out.push_str(&format!("simulation (testing): {}\n", self.simulated_families.join(", ")));
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
    run_prepared(Prepared { module, name: name.to_string(), from_bytecode: true }, opts)
}

fn run_prepared(prepared: Prepared, opts: &RunOptions) -> Result<RunReport, KazeError> {
    let module = prepared.module;
    let declared = module.needs.clone();
    // RB-08 (AUDIT-2026-11-08): every `--cap` backend is judged at GRANT time,
    // before the script runs, so `--cap sensor=bogus` fails as E_CAP_INVALID
    // instead of being accepted and then answered by the simulator.
    for (family, backend) in &opts.backend {
        if let Err(d) = crate::rt::hardware::validate_grant(family, backend) {
            return Err(KazeError::new(
                d.code,
                format!("{}: {} ({})", prepared.name, d.reason, d.rule),
            ));
        }
    }
    let simulated_families: Vec<String> = opts.backend.iter()
        .filter(|(_, backend)| backend == "sim")
        .map(|(family, _)| caps::family_of(family).to_string())
        .collect();
    // The old live host opened hardware before the VM's policy gate and built
    // its own hardware policy. Keep those paths unavailable until the caller's
    // policy and physical safety checks are propagated and verified.
    let mut blocked_live: Vec<String> = opts.backend.iter()
        .filter(|(family, backend)| matches!(caps::family_of(family), "drone" | "rover" | "sensor" | "device")
            && !backend.is_empty() && backend != "sim")
        .map(|(family, _)| caps::family_of(family).to_string())
        .collect();
    if !opts.pins.is_empty() {
        blocked_live.push("device".to_string());
    }
    let base_host = SimHost::with_configuration(
        opts.sim.clone(), simulated_families.clone(), blocked_live,
    );
    let use_ephemeral_memory = opts.backend.iter().any(|(family, backend)|
        caps::family_of(family) == "memory" && backend == "ephemeral");
    let mut host: Box<dyn CapHost> = if use_ephemeral_memory {
        Box::new(crate::rt::kaze_memory::EphemeralMemoryHost::new(base_host))
    } else {
        Box::new(base_host)
    };
    let limits = Limits { fuel: opts.fuel.max(1), heap_bytes: opts.heap_bytes, ..Limits::default() };

    let entry = resolve_entry(&module, opts)?;
    if entry.is_none() {
        // Hooks-only programs have no entry parameters, but an explicitly
        // selected or malformed default config must still fail before hooks.
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
        simulated_families,
    };

    {
        let mut vm = Vm::new(&module, &mut *host, &declared, &opts.grants, opts.policy.as_ref(), limits);
        vm.set_runtime_policy(opts.runtime_policy.as_ref());

        if let Some(name) = &entry {
            let item = module
                .item(name)
                .ok_or_else(|| KazeError::new("E_KAZE_ENTRY", format!("there is no '{}' in this script", name)))?;
            let merged = layered_args(opts, item.params.as_slice())?;
            let args = bind_args(item.params.as_slice(), &merged, &mut vm)
                .map_err(|e| KazeError::new(&e.code, format!("{}: {}", prepared.name, e.message)))?;
            let value = vm
                .run_item(item, &args)
                .map_err(|e| {
                    let mut err = KazeError::at(
                        e.code,
                        format!("{}: {} (line {})", prepared.name, e.msg, e.line),
                        &prepared.name,
                        e.line,
                        None,
                    );
                    if let Some(loc) = &mut err.location {
                        loc.routine_start = item.line_map.is_empty() && e.line != 0 && e.line == item.line;
                        loc.bytecode_source = prepared.from_bytecode;
                    }
                    err
                })?;
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
                    let mut err = KazeError::at(
                        e.code,
                        format!("{}: {} (line {})", prepared.name, e.msg, e.line),
                        &prepared.name,
                        e.line,
                        None,
                    );
                    if let Some(loc) = &mut err.location {
                        loc.routine_start = item.line_map.is_empty() && e.line != 0 && e.line == item.line;
                        loc.bytecode_source = prepared.from_bytecode;
                    }
                    err
                })?;
            }
            report.ticks += 1;
        }

        report.fuel_used = vm.fuel_used;
        report.instructions = vm.instructions;
        // KAZE-SPEC §8 says a run's memory is released in one operation when it
        // ends. Do exactly that, here, so the documented mechanism is the one
        // that runs rather than a side effect of dropping the Vm.
        vm.heap.free_wholesale();
        vm.arena.free_wholesale();
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

/// Bind the entry routine's parameters, refusing a run that is missing one.
///
/// KZ-035 (AUDIT-2026-11-08): an unbound parameter used to become nothing,
/// so the script ran on a silent hole (and often died later on an arithmetic
/// crash far from the cause). Every parameter without a value is now named
/// up front, with the three ways to supply it.
fn bind_args(
    params: &[String],
    args: &serde_json::Map<String, serde_json::Value>,
    vm: &mut Vm,
) -> Result<Vec<value::Value>, KazeError> {
    let missing: Vec<&String> = params
        .iter()
        .enumerate()
        .filter(|(i, p)| args.get(*p).is_none() && args.get(&i.to_string()).is_none())
        .map(|(_, p)| p)
        .collect();
    if !missing.is_empty() {
        return Err(KazeError::new("E_KAZE_PARAM", missing_param_message(&missing)));
    }
    let mut out = Vec::with_capacity(params.len());
    for (i, p) in params.iter().enumerate() {
        let json = args.get(p).or_else(|| args.get(&i.to_string())).expect("checked above");
        out.push(json_to_value(json, &mut vm.arena, &mut vm.heap)
            .map_err(|e| KazeError::new("E_KAZE_MEMORY", e))?);
    }
    Ok(out)
}

/// The E_KAZE_PARAM message: which parameter is missing and the three tiers
/// that could have supplied it (CLI > AGENT_<NAME> > config file).
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

/// Merge the parameter tiers for one run, lowest first: ./agent.config.json,
/// then AGENT_<NAME> / SOFUU_<NAME>, then the CLI arguments already parsed into
/// opts.args (CP-037, KZ-035).
///
/// The default file is optional; explicitly selected files and malformed or
/// unreadable existing files fail instead of silently changing configuration.
fn layered_args(
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
            let parsed: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| KazeError::new("E_KAZE_ARGS", format!("invalid configuration '{}': {}", config_path, e)))?;
            let mut cfg = parsed.as_object().cloned()
                .ok_or_else(|| KazeError::new("E_KAZE_ARGS", format!("configuration '{}' must be an object", config_path)))?;
            let map = match cfg.remove("args") {
                Some(serde_json::Value::Object(args)) => args,
                Some(_) => return Err(KazeError::new("E_KAZE_ARGS", format!("configuration '{}' args must be an object", config_path))),
                None => cfg,
            };
            for (k, v) in map {
                out.insert(k, v);
            }
        }
        Err(e) if opts.config_path.is_none() && e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(KazeError::new("E_KAZE_IO", format!("cannot read configuration '{}': {}", config_path, e))),
    }
    for p in params {
        let env_name = p.to_ascii_uppercase().replace('-', "_");
        for key in [format!("AGENT_{env_name}"), format!("SOFUU_{env_name}")] {
            if let Ok(raw) = std::env::var(&key) {
                let v = serde_json::from_str(&raw)
                    .unwrap_or(serde_json::Value::String(raw));
                out.insert(p.clone(), v);
                break;
            }
        }
    }
    // CLI positionals are in opts.args under numeric keys. Resolve them to
    // parameter names before overlaying named CLI arguments, so neither the
    // environment nor config file can shadow a positional value.
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

pub(crate) fn json_to_value(
    j: &serde_json::Value,
    arena: &mut arena::Arena,
    heap: &mut value::Heap,
) -> Result<value::Value, String> {
    use serde_json::Value as J;
    Ok(match j {
        J::Null => value::Value::Unit,
        J::Bool(b) => value::Value::Bool(*b),
        J::Number(n) => value::Value::Num(n.as_f64().filter(|v| v.is_finite())
            .ok_or_else(|| "number cannot be represented as a finite Kaze value".to_string())?),
        J::String(s) => value::Value::Str(arena.alloc(s)?),
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
                let key = value::Value::Str(arena.alloc(k)?);
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
unsafe fn js_source_of(ctx: *mut JSContext, input: JSValueConst, text: &str) -> Result<(String, String), KazeError> {
    if text.contains('\n') {
        return Ok((text.to_string(), "<inline>".to_string()));
    }
    let named_script = text.ends_with(".kaze") || text.ends_with(".kz");
    // Resolve with the same native root and active filesystem policy used by
    // sofuu.fs before checking existence or opening an embedded script path.
    let path = match crate::rt::fs::fs_path_arg(ctx, input, "kaze", crate::rt::policy::FsKind::Read) {
        Ok(path) => path,
        Err(_) => {
            // fs_path_arg throws; this API returns a structured report instead.
            let exception = qjs::JS_GetException(ctx);
            let code_value = qjs::sofuu_js_get_property_str(ctx, exception, c"code".as_ptr());
            let code_text = js_text(ctx, code_value);
            qjs::sofuu_js_free_value(ctx, code_value);
            let message_value = qjs::sofuu_js_get_property_str(ctx, exception, c"message".as_ptr());
            let message = js_text(ctx, message_value);
            qjs::sofuu_js_free_value(ctx, message_value);
            qjs::sofuu_js_free_value(ctx, exception);
            // One-line inline source remains usable even when file reads are
            // forbidden. It never causes a stat or open on this branch.
            if !named_script && text.chars().any(char::is_whitespace) {
                return Ok((text.to_string(), "<inline>".to_string()));
            }
            let code = match code_text.as_str() {
                "E_PATH_OUTSIDE" => "E_PATH_OUTSIDE",
                "E_PERMISSION_DENIED" => "E_PERMISSION_DENIED",
                _ => "E_CAP_DENIED",
            };
            return Err(KazeError::new(code, format!("script file access denied: {message}")));
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

fn report_to_js(ctx: *mut JSContext, report: &RunReport) -> JSValue {
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
            let v = qjs::sofuu_js_new_string(ctx, CString::new(family.as_str()).unwrap_or_default().as_ptr());
            qjs::JS_SetPropertyUint32(ctx, simulated, i as u32, v);
        }
        qjs::sofuu_js_set_property_str(ctx, obj, c"simulated_families".as_ptr(), simulated);
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
        let mut diagnostic = err.diagnostic(None).to_json().to_string();
        let len = diagnostic.len();
        // QuickJS requires a NUL at buf[len], even though len is explicit.
        // A bare Rust String can leave unrelated bytes there, turning valid
        // diagnostics into an intermittent parse exception.
        diagnostic.push('\0');
        let value = qjs::JS_ParseJSON(ctx, diagnostic.as_ptr().cast(), len, c"<kaze-diagnostic>".as_ptr());
        if qjs::is_exception(value) {
            qjs::sofuu_js_free_value(ctx, obj);
            return value;
        }
        qjs::sofuu_js_set_property_str(ctx, obj, c"diagnostic".as_ptr(), value);
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

fn opts_from_json(j: Option<serde_json::Value>) -> Result<RunOptions, KazeError> {
    let mut opts = RunOptions::default();
    let map = match j {
        None | Some(serde_json::Value::Null) => return Ok(opts),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => return Err(KazeError::new("E_KAZE_ARGS", "run options must be an object")),
    };
    if let Some(entry) = map.get("entry") {
        opts.entry = Some(entry.as_str().ok_or_else(|| KazeError::new("E_KAZE_ARGS", "entry must be text"))?.to_string());
    }
    if let Some(fuel) = map.get("fuel") {
        opts.fuel = fuel.as_u64().ok_or_else(|| KazeError::new("E_KAZE_ARGS", "fuel must be a nonnegative integer"))?;
    }
    if let Some(bytes) = map.get("heapBytes") {
        let bytes = bytes.as_u64().ok_or_else(|| KazeError::new("E_KAZE_ARGS", "heapBytes must be a nonnegative integer"))?;
        opts.heap_bytes = usize::try_from(bytes).map_err(|_| KazeError::new("E_KAZE_ARGS", "heapBytes is too large"))?;
    }
    if let Some(ticks) = map.get("ticks") {
        let ticks = ticks.as_u64().filter(|n| *n <= 10_000)
            .ok_or_else(|| KazeError::new("E_KAZE_ARGS", "ticks must be an integer from 0 to 10000"))?;
        opts.ticks = ticks as u32;
    }
    if let Some(ms) = map.get("tickMs") {
        opts.tick_ms = ms.as_u64().filter(|n| *n > 0)
            .ok_or_else(|| KazeError::new("E_KAZE_ARGS", "tickMs must be a positive integer"))?;
    }
    let add_grant = |opts: &mut RunOptions, raw: &str, explicit_backend: Option<&str>| -> Result<(), KazeError> {
        let (family, backend) = if let Some(backend) = explicit_backend {
            (raw.trim(), backend.trim())
        } else if let Some((family, backend)) = raw.split_once('=') {
            if backend.trim().is_empty() {
                return Err(KazeError::new("E_CAP_INVALID", "a backend name cannot be empty"));
            }
            (family.trim(), backend.trim())
        } else {
            (raw.trim(), "")
        };
        if family.is_empty() {
            return Err(KazeError::new("E_CAP_INVALID", "a capability grant needs a family"));
        }
        crate::rt::hardware::validate_grant(family, backend)
            .map_err(|d| KazeError::new(d.code, d.reason))?;
        if !opts.grants.iter().any(|g| g == family) {
            opts.grants.push(family.to_string());
        }
        if !backend.is_empty() || !opts.backend.iter().any(|(f, _)| f == family) {
            opts.backend.retain(|(f, _)| f != family);
            opts.backend.push((family.to_string(), backend.to_string()));
        }
        Ok(())
    };
    for key in ["caps", "grants"] {
        match map.get(key) {
            Some(serde_json::Value::Object(o)) => {
                for (family, backend) in o {
                    match backend {
                        serde_json::Value::String(value) if !value.trim().is_empty() => add_grant(&mut opts, family, Some(value))?,
                        serde_json::Value::Bool(true) | serde_json::Value::Null => add_grant(&mut opts, family, None)?,
                        _ => return Err(KazeError::new("E_CAP_INVALID", format!("{} backend for '{}' must be text or true", key, family))),
                    }
                }
            }
            Some(serde_json::Value::Array(a)) => {
                for grant in a {
                    let grant = grant.as_str().ok_or_else(|| KazeError::new("E_CAP_INVALID", format!("{} entries must be text", key)))?;
                    add_grant(&mut opts, grant, None)?;
                }
            }
            None => {}
            _ => return Err(KazeError::new("E_CAP_INVALID", format!("{} must be an array or object", key))),
        }
    }
    if map.contains_key("policy") && map.contains_key("policyJson") {
        return Err(KazeError::new("E_CAP_DENIED", "specify policy or policyJson, not both"));
    }
    let parsed_policy = if let Some(raw) = map.get("policyJson") {
        let raw = raw.as_str().ok_or_else(|| KazeError::new("E_CAP_DENIED", "policyJson must be text"))?;
        Some(crate::rt::policy::parse_policy_strict(raw)
            .map_err(|e| KazeError::new("E_CAP_DENIED", format!("invalid policy JSON: {}", e)))?)
    } else {
        map.get("policy").cloned()
    };
    if let Some(policy) = parsed_policy {
        crate::rt::policy::validate_policy(&policy)
            .map_err(|e| KazeError::new("E_CAP_DENIED", format!("invalid policy: {}", e)))?;
        opts.policy = Some(policy);
    }
    if let Some(pins) = map.get("pins") {
        let pins = pins.as_object().ok_or_else(|| KazeError::new("E_KAZE_ARGS", "pins must be an object"))?;
        for (name, value) in pins {
            let line = value.as_u64().and_then(|n| u16::try_from(n).ok())
                .ok_or_else(|| KazeError::new("E_KAZE_ARGS", format!("pin '{}' must be an unsigned 16-bit integer", name)))?;
            opts.pins.push((name.clone(), line));
        }
    }
    if let Some(value) = map.get("sim") {
        let sim = value.as_object().ok_or_else(|| KazeError::new("E_KAZE_ARGS", "sim must be an object"))?;
        if !opts.backend.iter().any(|(family, backend)| backend == "sim"
            && matches!(caps::family_of(family), "drone" | "rover")) {
            return Err(KazeError::new("E_CAP_INVALID", "sim requires an explicit drone=sim or rover=sim testing backend"));
        }
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
        opts.sim.apply_overrides(&spec).map_err(|e| KazeError::new("E_CAP_INVALID", e))?;
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
            let err = KazeError::new("E_KAZE_ARGS", "sfs.run(pathOrSource, args?, opts?) needs a script");
            return error_to_js(ctx, &err);
        }
        let text = js_text(ctx, *argv);
        let args = if argc > 1 && !qjs::is_undefined(*argv.add(1)) && !qjs::is_null(*argv.add(1)) {
            match js_json(ctx, *argv.add(1)) {
                Some(serde_json::Value::Object(m)) => m,
                _ => return error_to_js(ctx, &KazeError::new("E_KAZE_ARGS", "routine arguments must be a JSON-compatible object")),
            }
        } else {
            serde_json::Map::new()
        };
        let opts_json = if argc > 2 && !qjs::is_undefined(*argv.add(2)) && !qjs::is_null(*argv.add(2)) {
            match js_json(ctx, *argv.add(2)) {
                Some(value) => Some(value),
                None => return error_to_js(ctx, &KazeError::new("E_KAZE_ARGS", "run options must be valid JSON-compatible values")),
            }
        } else { None };
        let mut opts = match opts_from_json(opts_json) {
            Ok(opts) => opts,
            Err(err) => return error_to_js(ctx, &err),
        };
        opts.args = args;
        opts.ambient_args = false;
        opts.runtime_policy = Some(crate::rt::policy::policy_json(ctx));

        let result = js_source_of(ctx, *argv, &text).and_then(|(src, name)| run_source(&src, &name, &opts));
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
        let result = js_source_of(ctx, *argv, &text).and_then(|(src, name)| check_source(&src, &name));
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
        let result = js_source_of(ctx, *argv, &text).and_then(|(src, name)| prepare(&src, &name));
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
        let gate = qjs::sofuu_js_new_string(ctx, c"fail-closed-1".as_ptr());
        qjs::sofuu_js_set_property_str(ctx, ns, c"capabilityGate".as_ptr(), gate);
        let boundary = qjs::sofuu_js_new_string(ctx, c"fs-and-args-1".as_ptr());
        qjs::sofuu_js_set_property_str(ctx, ns, c"embeddedBoundary".as_ptr(), boundary);
        qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"sfs".as_ptr(), qjs::sofuu_js_dup_value(ctx, ns));
        qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"kaze".as_ptr(), ns);
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

    #[test]
    fn js_kaze_failure_exposes_the_diagnostic_contract() {
        let _guard = crate::rt::test_loop_lock();
        let rt = sofuu_ffi::SofuuRuntime::init().expect("runtime init");
        let rc = rt.eval_string(
            r#"
            const source = 'needs: policy\nfilter main():\n    let z = 0\n    let y = 1 / z\n    allow "ok"\n';
            const result = sofuu.kaze.run(source, {}, {grants: ['policy']});
            if (result.ok || result.code !== 'E_KAZE_DIV0') throw new Error('missing failure');
            if (!result.diagnostic || result.diagnostic.code !== result.code ||
                result.diagnostic.location.line !== 4) throw new Error('missing location');
            if (!sofuu.diagnostics.render(JSON.stringify(result.diagnostic)).includes('error[E_KAZE_DIV0]'))
                throw new Error('not rendered');
            "#,
            "<kaze-diagnostic-test>",
        );
        assert_eq!(rc, 0);
    }

    #[test]
    fn vm_diagnostics_keep_statement_lines_across_bytecode_versions() {
        let src = "needs: policy\nfilter main():\n    if true:\n        let z = 0\n        let y = 1 / z\n    allow \"ok\"\n";
        let opts = RunOptions::default();
        let err = run_source(src, "agent.kaze", &opts).unwrap_err();
        assert_eq!(err.code, "E_KAZE_DIV0");
        assert_eq!(err.diagnostic(None).location.unwrap().line, Some(5));

        let prepared = prepare(src, "agent.kaze").unwrap();
        let mut bytes = isa::serialize(&prepared.module);
        let result = run_module(isa::deserialize(&bytes).unwrap(), "agent.kazebc", &opts).unwrap_err();
        let location = result.diagnostic(None).location.unwrap();
        assert_eq!(location.line, Some(5));
        assert!(location.label.unwrap().contains("bytecode"));

        // This fixture has one item, whose v2 line table is last in the file.
        let nlines = prepared.module.items[0].line_map.len();
        bytes.truncate(bytes.len() - 4 - nlines * 4);
        bytes[8..10].copy_from_slice(&1u16.to_le_bytes());
        let result = run_module(isa::deserialize(&bytes).unwrap(), "legacy.kazebc", &opts).unwrap_err();
        let location = result.diagnostic(None).location.unwrap();
        assert_eq!(location.line, Some(2));
        assert!(location.label.unwrap().contains("routine start"));
    }

    #[test]
    fn lists_compare_by_content_not_by_identity() {
        // Two separately built lists with the same items are equal, matching how
        // strings already behave. Before this, equality was the heap index, so
        // this returned 0 while "a" == "a" returned 1.
        let src = "needs: log\naction main():\n    let a = [1, 2]\n    let b = [1, 2]\n    \
                   if a == b:\n        return 1\n    return 0\n";
        let report = run_source(src, "<eq>", &RunOptions::default()).expect("runs");
        assert_eq!(report.value, "1");

        let src = "needs: log\naction main():\n    let a = [1, 2]\n    let b = [1, 3]\n    \
                   if a == b:\n        return 1\n    return 0\n";
        let report = run_source(src, "<ne>", &RunOptions::default()).expect("runs");
        assert_eq!(report.value, "0");
    }

    #[test]
    fn a_missing_parameter_keeps_all_three_configuration_tiers() {
        let err = run_source(
            "needs: log\naction main(question):\n    log question\n",
            "<param>",
            &RunOptions::default(),
        )
        .expect_err("a required parameter with no value must be refused");
        assert_eq!(err.code, "E_KAZE_PARAM");
        let diag = err.diagnostic(None);
        /* The guidance is deliberately multi-line in the legacy message; the
         * diagnostic flattens it, but must not drop tiers. Only the checker's
         * multi-error message is truncated to its first line. */
        for tier in ["CLI flag", "AGENT_QUESTION", "--config"] {
            assert!(
                diag.message.contains(tier),
                "missing {tier} in {:?}",
                diag.message
            );
        }
    }

    #[test]
    fn a_wholesale_free_gives_the_whole_budget_back() {
        let mut heap = value::Heap::new(256);
        heap.push_array(vec![value::Value::Num(1.0); 4]).expect("fits");
        assert!(heap.used_bytes() > 0);
        assert!(heap.used_bytes() <= 256);
        heap.free_wholesale();
        assert_eq!(heap.used_bytes(), 0, "the meter must be part of the free");
    }
    use super::*;

    fn run(src: &str, opts: &RunOptions) -> RunReport {
        run_source(src, "<test>", opts).unwrap_or_else(|e| panic!("run failed: {}", e.render()))
    }

    fn grants(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn test_policy() -> Option<serde_json::Value> {
        Some(serde_json::json!({"tools": {"allow": ["kaze.*"]}}))
    }

    #[test]
    fn a_patrol_runs_end_to_end_with_a_deterministic_trace() {
        let src = "needs: drone, log\n\naction patrol(points):\n    log \"Starting patrol\"\n    arm drone\n    takeoff to 15 meters\n    for each point in points:\n        fly to point at 15 meters\n    return to launch\n";
        let mut args = serde_json::Map::new();
        args.insert("points".into(), serde_json::json!([1, 2, 3]));
        let opts = RunOptions { args, grants: grants(&["drone", "log"]), backend: vec![("drone".into(), "sim".into())], policy: test_policy(), ..Default::default() };
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
        let src = "needs: rover\n\ncheck scan():\n    let d = rover.telemetry(\"distance\")\n    if d < 1 meters:\n        stop rover\n    otherwise:\n        drive forward at 0.5 m/s\n";
        let opts = RunOptions { grants: grants(&["rover"]), backend: vec![("rover".into(), "sim".into())], policy: test_policy(), ..Default::default() };
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
        assert!(err.message.contains("--cap drone"), "{}", err.message);
    }

    #[test]
    fn fuel_stops_a_runaway_loop() {
        let src = "needs: log\naction main():\n    while true:\n        log \"spin\"\n";
        let opts = RunOptions { fuel: 500, grants: grants(&["log"]), policy: test_policy(), ..Default::default() };
        let err = run_source(src, "<test>", &opts).unwrap_err();
        assert_eq!(err.code, "E_TIMEOUT");
    }

    #[test]
    fn the_low_battery_hook_fires_only_below_the_threshold() {
        let src = "needs: drone, log\n\naction main():\n    log \"patrol\"\n\nwhen battery < 20%:\n    log warning \"Battery critical\"\n    return to launch\n";
        let mut sim = SimState::default();
        sim.battery = 15.0;
        let opts = RunOptions { grants: grants(&["drone", "log"]), backend: vec![("drone".into(), "sim".into())], policy: test_policy(), ticks: 1, sim, ..Default::default() };
        let hot = run(src, &opts);
        assert!(hot.actions.iter().any(|a| a.contains("drone.rtl")), "{:?}", hot.actions);

        let opts2 = RunOptions { grants: grants(&["drone", "log"]), backend: vec![("drone".into(), "sim".into())], policy: test_policy(), ticks: 1, ..Default::default() };
        let cold = run(src, &opts2);
        assert!(!cold.actions.iter().any(|a| a.contains("drone.rtl")), "{:?}", cold.actions);
    }

    #[test]
    fn every_hundred_ms_fires_on_a_hundred_ms_tick_grid() {
        let src = "needs: sensor, log\n\nevery 100ms:\n    log \"tick\"\n";
        let opts = RunOptions { grants: grants(&["log"]), policy: test_policy(), ticks: 3, tick_ms: 100, ..Default::default() };
        let report = run(src, &opts);
        assert_eq!(report.logs.len(), 3);
        assert_eq!(report.ticks, 3);
    }

    #[test]
    fn a_guardrail_returns_a_verdict_instead_of_a_side_effect() {
        let src = "needs: policy\n\nfilter user_prompt(prompt):\n    if length of prompt > 20:\n        reject with \"too long\"\n    allow prompt\n";
        let mut args = serde_json::Map::new();
        args.insert("prompt".into(), serde_json::json!("hello"));
        let report = run(src, &RunOptions { args, grants: grants(&["policy"]), policy: test_policy(), ..Default::default() });
        assert_eq!(report.verdict, Some(Verdict::Allow("\"hello\"".to_string())));

        let mut args = serde_json::Map::new();
        args.insert("prompt".into(), serde_json::json!("this prompt is definitely too long"));
        let report = run(src, &RunOptions { args, grants: grants(&["policy"]), policy: test_policy(), ..Default::default() });
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
        let opts = RunOptions { grants: grants(&["drone"]), backend: vec![("drone".into(), "sim".into())], policy: test_policy(), ..Default::default() };
        let err = run_source(src, "<test>", &opts).unwrap_err();
        assert_eq!(err.code, "E_KAZE_RANGE");
    }

    #[test]
    fn a_record_argument_is_read_by_field() {
        // Maps have no literal syntax in v0.1: they arrive as arguments, and
        // `record.field` is how a script reads one.
        let src = "action main(record):\n    return record.level\n";
        let mut args = serde_json::Map::new();
        args.insert("record".into(), serde_json::json!({"kind": "fire", "level": 3}));
        let opts = RunOptions { args, ..Default::default() };
        let report = run(src, &opts);
        assert_eq!(report.value, "3");
        assert!(report.actions.is_empty());

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
        let opts = RunOptions { grants: grants(&["drone"]), backend: vec![("drone".into(), "sim".into())], policy: test_policy(), ..Default::default() };
        let report = run(src, &opts);
        assert!(
            report.actions.iter().any(|a| a == "drone.goto(1, 2, 3) -> nothing"),
            "{:?}",
            report.actions
        );
    }

    #[test]
    fn a_named_sensor_without_a_backend_is_unavailable() {
        let src = "needs: sensor\naction main():\n    let t = sensor.read(\"temperature\")\n    return t\n";
        let opts = RunOptions { grants: grants(&["sensor"]), policy: test_policy(), ..Default::default() };
        assert_eq!(run_source(src, "<test>", &opts).unwrap_err().code, "E_CAP_MISSING");
    }

    /// RB-08 (AUDIT-2026-11-08): `--cap sensor=bogus` was accepted at grant
    /// time, the run proceeded, and the simulator answered for a backend nobody
    /// had implemented. The grant is judged here, before the script runs.
    #[test]
    fn a_bogus_backend_grant_is_refused_before_the_script_runs() {
        let src = "needs: sensor\naction main():\n    let t = sensor.read(\"temperature\")\n    return t\n";
        let opts = RunOptions {
            grants: grants(&["sensor"]),
            backend: vec![("sensor".to_string(), "bogus".to_string())],
            ..Default::default()
        };
        let err = run_source(src, "<grant>", &opts).unwrap_err();
        assert_eq!(err.code, "E_CAP_INVALID", "{}", err.message);
        assert!(err.message.contains("bogus"), "{}", err.message);
        // Sensor simulation is not an allowed standalone sensor backend.
        let opts = RunOptions {
            grants: grants(&["sensor"]),
            backend: vec![("sensor".to_string(), "sim".to_string())],
            ..Default::default()
        };
        assert_eq!(run_source(src, "<grant>", &opts).unwrap_err().code, "E_CAP_INVALID");
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

    #[test]
    fn unavailable_ai_backend_fails_instead_of_fabricating_a_response() {
        let src = "needs: ai\naction main():\n    return ai.prompt(\"hello\")\n";
        let opts = RunOptions {
            grants: vec!["ai".into()],
            policy: Some(serde_json::json!({"tools": {"deny": []}})),
            ..Default::default()
        };
        let err = run_source(src, "<test_agent>", &opts).unwrap_err();
        assert_eq!(err.code, "E_CAP_MISSING");
        assert!(err.message.contains("ai.prompt"));
    }
}
