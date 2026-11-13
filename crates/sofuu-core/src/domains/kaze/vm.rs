// sfs/vm.rs — the bytecode interpreter (M-K3).
//
// PLAN-KAZE-LANGUAGE §1.4/§1.5/§4. Two guarantees are enforced here and
// nowhere else:
//
//   * Fuel. Every instruction costs one unit. At zero the run stops with
//     E_TIMEOUT (fuel) — not a panic, not a hang, not a thread to kill. `FUEL_TICK` at
//     each loop head also polls the host deadline (rt/deadline.rs), so a run
//     started from JS obeys the same interrupt as any other host call.
//   * One side-effect path. CALL_CAP is the only instruction that can reach
//     the outside world, and it always asks rt/hardware.rs first. A Kaze
//     program cannot open a socket because there is no instruction for it.
//
// Arithmetic traps instead of wrapping: a non-finite result is E_KAZE_OVERFLOW,
// a division by zero is E_KAZE_DIV0, and a bad index is E_KAZE_BOUNDS. In a
// controller, a silently-wrong number is worse than a stopped run.

use crate::sfs::arena::{fmt_num, Arena};
use crate::sfs::caps::{self, CapHost};
use crate::sfs::isa::{field_a, opcode, Const, ItemCode, Module, Op};
use crate::sfs::value::{Heap, Value};
use std::sync::atomic::{AtomicU64, Ordering};

/// How much work one run may do before it is stopped.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub fuel: u64,
    pub arena_bytes: usize,
    /// Budget for arrays and maps. Two meters, not one: fuel bounds work,
    /// this bounds retained bytes.
    pub heap_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        // 200k instructions is a few milliseconds of work and far more than any
        // hardware routine needs; the point is that the bound exists.
        Limits { fuel: 200_000, arena_bytes: 256 * 1024, heap_bytes: Heap::DEFAULT_LIMIT }
    }
}

#[derive(Debug, Clone)]
pub struct VmError {
    pub code: &'static str,
    pub msg: String,
    pub line: u32,
}

impl VmError {
    fn new(code: &'static str, msg: impl Into<String>) -> Self {
        VmError { code, msg: msg.into(), line: 0 }
    }
}

/// What a `filter` decided.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    Allow(String),
    Reject(String),
}

/// A suspended item owns its bytecode and exact program counter. Host calls
/// still complete synchronously; instruction slices are the scheduling
/// foundation for an asynchronous driver.
pub(crate) struct ItemContinuation {
    item: ItemCode,
    pc: usize,
    token: u64,
    finished: bool,
    pending: Option<PendingCapability>,
}

struct PendingCapability {
    id: u64,
    line: u32,
    family: String,
    op: String,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CapabilityRequest {
    pub id: u64,
    pub family: String,
    pub op: String,
    pub args: serde_json::Value,
    pub line: u32,
}

#[derive(Debug, PartialEq)]
pub(crate) enum SliceOutcome {
    Yielded,
    Capability(CapabilityRequest),
    Complete(Value),
}

// Tokens prevent a continuation from being used with a different VM or after
// its VM has started another item. Zero is reserved for exhaustion.
static NEXT_CONTINUATION_TOKEN: AtomicU64 = AtomicU64::new(1);
static NEXT_CAPABILITY_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
const MAX_CAPABILITY_JSON_BYTES: usize = 1024 * 1024;

enum Host<'a> {
    Borrowed(&'a mut dyn CapHost),
    Owned(Box<dyn CapHost + 'a>),
}

impl CapHost for Host<'_> {
    fn call(
        &mut self,
        family: &str,
        op: &str,
        args: &[Value],
        arena: &mut Arena,
        heap: &mut Heap,
    ) -> Result<Value, String> {
        match self {
            Host::Borrowed(host) => host.call(family, op, args, arena, heap),
            Host::Owned(host) => host.call(family, op, args, arena, heap),
        }
    }

    fn actions(&self) -> &[crate::sfs::caps::Action] {
        match self {
            Host::Borrowed(host) => host.actions(),
            Host::Owned(host) => host.actions(),
        }
    }

    fn logs(&self) -> &[(String, String)] {
        match self {
            Host::Borrowed(host) => host.logs(),
            Host::Owned(host) => host.logs(),
        }
    }

    fn check_async_dispatch(&self) -> Result<(), String> {
        match self {
            Host::Borrowed(host) => host.check_async_dispatch(),
            Host::Owned(host) => host.check_async_dispatch(),
        }
    }

    fn note_async_completion(&mut self, family: &str, op: &str, succeeded: bool) {
        match self {
            Host::Borrowed(host) => host.note_async_completion(family, op, succeeded),
            Host::Owned(host) => host.note_async_completion(family, op, succeeded),
        }
    }

    fn note_async_unknown(&mut self, family: &str, op: &str) {
        match self {
            Host::Borrowed(host) => host.note_async_unknown(family, op),
            Host::Owned(host) => host.note_async_unknown(family, op),
        }
    }
}

pub struct Vm<'a> {
    pub module: Module,
    pub arena: Arena,
    pub heap: Heap,
    host: Host<'a>,
    declared: Vec<String>,
    grants: Vec<String>,
    policy: Option<serde_json::Value>,
    runtime_policy: Option<serde_json::Value>,
    fuel_left: u64,
    pub fuel_used: u64,
    pub instructions: u64,
    stack: Vec<Value>,
    locals: Vec<Value>,
    verdict: Option<Verdict>,
    active_item: Option<u64>,
}

/// The stack is a fixed-size vector; a script cannot grow it without bound.
const MAX_STACK: usize = 256;

impl<'a> Vm<'a> {
    pub fn new(
        module: &Module,
        host: &'a mut dyn CapHost,
        declared: &[String],
        grants: &[String],
        policy: Option<&serde_json::Value>,
        limits: Limits,
    ) -> Self {
        Self::build(
            module.clone(),
            Host::Borrowed(host),
            declared.to_vec(),
            grants.to_vec(),
            policy.cloned(),
            limits,
        )
    }

    /// Build a VM whose module, host and capability configuration are owned by
    /// the run. This form can safely live across asynchronous JS callbacks.
    pub(crate) fn new_owned(
        module: Module,
        host: Box<dyn CapHost + 'a>,
        declared: Vec<String>,
        grants: Vec<String>,
        policy: Option<serde_json::Value>,
        limits: Limits,
    ) -> Self {
        Self::build(module, Host::Owned(host), declared, grants, policy, limits)
    }

    fn build(
        module: Module,
        host: Host<'a>,
        declared: Vec<String>,
        grants: Vec<String>,
        policy: Option<serde_json::Value>,
        limits: Limits,
    ) -> Self {
        Vm {
            module,
            arena: Arena::new(limits.arena_bytes),
            heap: Heap::new(limits.heap_bytes),
            host,
            declared,
            grants,
            policy,
            runtime_policy: None,
            fuel_left: limits.fuel,
            fuel_used: 0,
            instructions: 0,
            stack: Vec::with_capacity(32),
            locals: Vec::new(),
            verdict: None,
            active_item: None,
        }
    }

    /// Add the embedding runtime's restrictions without replacing the
    /// Kaze-specific policy required for every capability call.
    pub fn set_runtime_policy(&mut self, policy: Option<&serde_json::Value>) {
        self.runtime_policy = policy.cloned();
    }

    /// Run one item to completion. `args` bind to `item.params` by position.
    pub fn run_item(&mut self, item: &ItemCode, args: &[Value]) -> Result<Value, VmError> {
        let mut state = self.start_item(item, args)?;
        match self.run_slice(&mut state, usize::MAX)? {
            SliceOutcome::Complete(value) => Ok(value),
            SliceOutcome::Capability(request) => {
                // This synchronous API refuses before a dispatcher sees the
                // request; it is not an uncertain external outcome.
                state.pending = None;
                self.cancel_item(&mut state)?;
                Err(VmError {
                    code: "E_CAP_MISSING",
                    msg: format!("{}.{} requires an asynchronous capability driver", request.family, request.op),
                    line: request.line,
                })
            }
            SliceOutcome::Yielded => {
                // Synchronous callers cannot continue beyond this slice.
                self.cancel_item(&mut state)?;
                Err(VmError::new("E_KAZE_STATE", "synchronous item exceeded the instruction slice"))
            }
        }
    }

    /// Begin one item without running any instructions. At most one item may
    /// be active in a VM, so its stack and locals cannot be overwritten while
    /// a continuation is suspended.
    pub(crate) fn start_item(
        &mut self,
        item: &ItemCode,
        args: &[Value],
    ) -> Result<ItemContinuation, VmError> {
        if self.active_item.is_some() {
            return Err(VmError::new("E_KAZE_STATE", "another item is already active"));
        }
        let token = NEXT_CONTINUATION_TOKEN.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| VmError::new("E_KAZE_STATE", "continuation identifiers exhausted"))?;
        self.stack.clear();
        self.locals = vec![Value::Unit; item.slots.max(item.params.len() as u8) as usize];
        for (i, a) in args.iter().enumerate().take(self.locals.len()) {
            self.locals[i] = *a;
        }
        self.verdict = None;
        self.active_item = Some(token);
        Ok(ItemContinuation { item: item.clone(), pc: 0, token, finished: false, pending: None })
    }

    /// Execute at most `budget` instructions, retaining the exact next program
    /// counter on yield. A failure terminates the item; it must never be
    /// resumed and replay a side effect whose outcome may be unknown.
    pub(crate) fn run_slice(
        &mut self,
        state: &mut ItemContinuation,
        budget: usize,
    ) -> Result<SliceOutcome, VmError> {
        self.validate_continuation(state)?;
        if budget == 0 {
            return Err(VmError::new("E_KAZE_STATE", "instruction slice must be nonzero"));
        }
        if state.pending.is_some() {
            return Err(VmError::new("E_KAZE_STATE", "resume the pending capability before running another slice"));
        }
        let result = self.run_slice_inner(state, budget);
        if !matches!(result, Ok(SliceOutcome::Yielded | SliceOutcome::Capability(_))) {
            state.finished = true;
            self.active_item = None;
        }
        result
    }

    /// Complete exactly one pending asynchronous capability. The request id
    /// binds the response to this continuation and rejects duplicates/late
    /// replies. Errors terminate the run so an uncertain effect cannot replay.
    pub(crate) fn resume_capability(
        &mut self,
        state: &mut ItemContinuation,
        request_id: u64,
        result: Result<serde_json::Value, VmError>,
    ) -> Result<(), VmError> {
        self.validate_continuation(state)?;
        let Some(pending) = state.pending.as_ref() else {
            return Err(VmError::new("E_KAZE_STATE", "there is no pending capability to resume"));
        };
        if pending.id != request_id {
            return Err(VmError::new("E_KAZE_STATE", "capability response is stale, duplicated, or belongs to another run"));
        }
        let pending_line = pending.line;
        let pending_family = pending.family.clone();
        let pending_op = pending.op.clone();
        state.pending = None;
        let value = match result {
            Ok(json) => {
                let bytes = match serde_json::to_vec(&json) {
                    Ok(bytes) => bytes,
                    Err(_) => {
                        state.finished = true;
                        self.active_item = None;
                        self.host.note_async_completion(&pending_family, &pending_op, false);
                        return Err(VmError { code: "E_KAZE_HOST", msg: "capability result is not valid JSON".into(), line: pending_line });
                    }
                };
                if bytes.len() > MAX_CAPABILITY_JSON_BYTES {
                    state.finished = true;
                    self.active_item = None;
                    self.host.note_async_completion(&pending_family, &pending_op, false);
                    return Err(VmError { code: "E_KAZE_MEMORY", msg: "capability result exceeded the 1 MiB limit".into(), line: pending_line });
                }
                match crate::sfs::json_to_value(&json, &mut self.arena, &mut self.heap) {
                    Ok(value) => value,
                    Err(message) => {
                        state.finished = true;
                        self.active_item = None;
                        self.host.note_async_completion(&pending_family, &pending_op, false);
                        return Err(VmError { code: "E_KAZE_MEMORY", msg: message, line: pending_line });
                    }
                }
            }
            Err(mut error) => {
                state.finished = true;
                self.active_item = None;
                if error.line == 0 {
                    error.line = pending_line;
                }
                if error.code == "E_KAZE_OUTCOME_UNKNOWN" {
                    self.host.note_async_unknown(&pending_family, &pending_op);
                } else {
                    self.host.note_async_completion(&pending_family, &pending_op, false);
                }
                return Err(error);
            }
        };
        if let Err(error) = self.push(value, pending_line) {
            state.finished = true;
            self.active_item = None;
            self.host.note_async_completion(&pending_family, &pending_op, false);
            return Err(error);
        }
        self.host.note_async_completion(&pending_family, &pending_op, true);
        Ok(())
    }

    pub(crate) fn cancel_item(&mut self, state: &mut ItemContinuation) -> Result<(), VmError> {
        self.validate_continuation(state)?;
        if let Some(pending) = state.pending.take() {
            self.host.note_async_unknown(&pending.family, &pending.op);
        }
        state.finished = true;
        self.active_item = None;
        Ok(())
    }

    pub(crate) fn has_pending_capability(&self, state: &ItemContinuation) -> bool {
        state.pending.is_some()
    }

    fn validate_continuation(&self, state: &ItemContinuation) -> Result<(), VmError> {
        if state.finished || self.active_item != Some(state.token) {
            return Err(VmError::new("E_KAZE_STATE", "item continuation is no longer active in this VM"));
        }
        Ok(())
    }

    fn run_slice_inner(
        &mut self,
        state: &mut ItemContinuation,
        budget: usize,
    ) -> Result<SliceOutcome, VmError> {
        let item = &state.item;

        let code = &item.code;
        let mut pc = state.pc;
        let mut steps = 0usize;
        while pc < code.len() {
            if steps == budget {
                state.pc = pc;
                return Ok(SliceOutcome::Yielded);
            }
            let w = code[pc];
            let current_line = item.line_map.get(pc).copied()
                .filter(|line| *line > 0).unwrap_or(item.line);
            let Some(op) = Op::from_u8(opcode(w)) else {
                return Err(VmError::new("E_KAZE_BADCODE", "this module has an unknown instruction"));
            };
            pc += 1;
            // A loop head carries its own source line in the immediate word, so
            // an exhausted budget blames the loop rather than the routine.
            let step_line = if matches!(op, Op::FuelTick) {
                code.get(pc).copied().unwrap_or(current_line)
            } else {
                current_line
            };
            self.charge(step_line)?;
            steps += 1;
            match op {
                Op::Const => {
                    let imm = self.imm(code, pc, current_line)?;
                    pc += 1;
                    let v = self.constant(imm as usize, current_line)?;
                    self.push(v, current_line)?;
                }
                Op::LoadLocal => {
                    let slot = field_a(w) as usize;
                    let v = *self
                        .locals
                        .get(slot)
                        .ok_or_else(|| VmError::new("E_KAZE_BADCODE", "LOAD from a missing slot"))?;
                    self.push(v, current_line)?;
                }
                Op::StoreLocal => {
                    let slot = field_a(w) as usize;
                    let v = self.pop(current_line)?;
                    if let Some(dst) = self.locals.get_mut(slot) {
                        *dst = v;
                    } else {
                        return Err(VmError::new("E_KAZE_BADCODE", "STORE to a missing slot"));
                    }
                }
                Op::NewArr => {
                    let n = field_a(w) as usize;
                    let mut items = Vec::with_capacity(n);
                    for _ in 0..n {
                        items.push(self.pop(current_line)?);
                    }
                    items.reverse();
                    let v = self
                        .heap
                        .push_array(items)
                        .map_err(|msg| VmError { code: "E_KAZE_MEMORY", msg, line: current_line })?;
                    self.push(v, current_line)?;
                }
                Op::GetIndex => {
                    let idx = self.pop(current_line)?;
                    let coll = self.pop(current_line)?;
                    let v = self.index(coll, idx, current_line)?;
                    self.push(v, current_line)?;
                }
                Op::Len => {
                    let v = self.pop(current_line)?;
                    let n = self.length(v, current_line)?;
                    self.push(Value::Num(n), current_line)?;
                }
                Op::GetField => {
                    let field = self.pop(current_line)?;
                    let base = self.pop(current_line)?;
                    let v = self.field(base, field, current_line)?;
                    self.push(v, current_line)?;
                }
                Op::Jeq | Op::Jne => {
                    let imm = self.imm(code, pc, current_line)? as i32;
                    pc += 1;
                    let b = self.pop(current_line)?;
                    let a = self.pop(current_line)?;
                    let eq = crate::sfs::arena::values_equal(a, b, &self.arena, &self.heap);
                    let taken = if op == Op::Jeq { eq } else { !eq };
                    if taken {
                        pc = jump(pc, imm, code.len(), current_line)?;
                    }
                }
                Op::Pop => {
                    self.pop(current_line)?;
                }
                Op::Add => {
                    let b = self.pop(current_line)?;
                    let a = self.pop(current_line)?;
                    let v = self.add(a, b, current_line)?;
                    self.push(v, current_line)?;
                }
                Op::Sub => self.arith(Op::Sub, current_line)?,
                Op::Mul => self.arith(Op::Mul, current_line)?,
                Op::Div => self.arith(Op::Div, current_line)?,
                Op::Mod => self.arith(Op::Mod, current_line)?,
                Op::Neg => {
                    let v = self.pop(current_line)?;
                    match v {
                        Value::Num(n) => self.push(Value::Num(-n), current_line)?,
                        other => {
                            return Err(VmError::new(
                                "E_KAZE_TYPE",
                                format!("cannot negate {}", other.type_name()),
                            ))
                        }
                    }
                }
                Op::CmpLt | Op::CmpLe | Op::CmpGt | Op::CmpGe => {
                    let b = self.pop(current_line)?;
                    let a = self.pop(current_line)?;
                    let v = self.compare(op, a, b, current_line)?;
                    self.push(Value::Bool(v), current_line)?;
                }
                Op::CmpEq | Op::CmpNe => {
                    let b = self.pop(current_line)?;
                    let a = self.pop(current_line)?;
                    let eq = crate::sfs::arena::values_equal(a, b, &self.arena, &self.heap);
                    self.push(Value::Bool(if op == Op::CmpEq { eq } else { !eq }), current_line)?;
                }
                Op::Not => {
                    let v = self.pop(current_line)?;
                    self.push(Value::Bool(!v.is_truthy()), current_line)?;
                }
                Op::Jmp => {
                    let imm = self.imm(code, pc, current_line)? as i32;
                    pc = jump(pc + 1, imm, code.len(), current_line)?;
                }
                Op::JmpIfFalse | Op::JmpIfTrue => {
                    let imm = self.imm(code, pc, current_line)? as i32;
                    pc += 1;
                    let v = self.pop(current_line)?;
                    let taken = if op == Op::JmpIfTrue { v.is_truthy() } else { !v.is_truthy() };
                    if taken {
                        pc = jump(pc, imm, code.len(), current_line)?;
                    }
                }
                Op::CallCap => {
                    let imm = self.imm(code, pc, current_line)? as usize;
                    pc += 1;
                    if let Some(request) = self.call_cap(imm, field_a(w) as usize, current_line)? {
                        state.pc = pc;
                        state.pending = Some(PendingCapability {
                            id: request.id,
                            line: request.line,
                            family: request.family.clone(),
                            op: request.op.clone(),
                        });
                        return Ok(SliceOutcome::Capability(request));
                    }
                }
                Op::PolicyStop => {
                    let v = self.pop(current_line)?;
                    return Ok(SliceOutcome::Complete(v));
                }
                Op::Ret => {
                    let v = self.pop(current_line)?;
                    return Ok(SliceOutcome::Complete(v));
                }
                Op::RetUnit => return Ok(SliceOutcome::Complete(Value::Unit)),
                Op::FuelTick => {
                    // The pre-charge above already paid for this tick; the
                    // immediate only names the loop for error reporting.
                    let line = self.imm(code, pc, current_line)?;
                    pc += 1;
                    // A JS host may have armed a deadline for this whole call.
                    if crate::rt::deadline::poll_expired() != 0 {
                        // rt/deadline.rs already recorded the firing; ask it for
                        // the report so a host sees one story, not two.
                        let detail = crate::rt::deadline::classify_interrupted("interrupted")
                            .unwrap_or_else(|| {
                                "E_TIMEOUT: the call was interrupted after its deadline elapsed"
                                    .to_string()
                            });
                        let msg = detail
                            .strip_prefix("E_TIMEOUT: ")
                            .unwrap_or(detail.as_str())
                            .to_string();
                        return Err(VmError { code: "E_TIMEOUT", msg, line });
                    }
                }
                Op::Trap => {
                    let imm = self.imm(code, pc, current_line)? as usize;
                    let msg = match self.module.consts.get(imm) {
                        Some(Const::Str(s)) => s.clone(),
                        _ => "the script stopped".to_string(),
                    };
                    return Err(VmError { code: "E_KAZE_TRAP", msg, line: current_line });
                }
                Op::Halt => return Ok(SliceOutcome::Complete(Value::Unit)),
            }
        }
        Ok(SliceOutcome::Complete(Value::Unit))
    }

    /// Take the verdict the last `run_item` produced, if it was a `filter`.
    pub fn take_verdict(&mut self) -> Option<Verdict> {
        self.verdict.take()
    }

    pub fn fuel_left(&self) -> u64 {
        self.fuel_left
    }

    pub(crate) fn actions(&self) -> &[crate::sfs::caps::Action] {
        self.host.actions()
    }

    pub(crate) fn logs(&self) -> &[(String, String)] {
        self.host.logs()
    }

    // ── helpers ──────────────────────────────────────────────────────
    fn charge(&mut self, line: u32) -> Result<(), VmError> {
        self.instructions += 1;
        self.fuel_used += 1;
        if self.fuel_left == 0 {
            // PLAN-KAZE §1.5/§4: a runaway script halts uncatchably and comes
            // back as a typed E_TIMEOUT. The message names the cause, because
            // "out of fuel" and "the host's deadline fired" need different
            // fixes even though both are timeouts to the caller.
            return Err(VmError {
                code: "E_TIMEOUT",
                msg: "this run used all of its fuel and was stopped (fuel, not wall time); \
                      a loop that never ends or too much work in one tick is the usual cause"
                    .to_string(),
                line,
            });
        }
        self.fuel_left -= 1;
        Ok(())
    }

    fn imm(&self, code: &[u32], pc: usize, line: u32) -> Result<u32, VmError> {
        code.get(pc)
            .copied()
            .ok_or_else(|| VmError { code: "E_KAZE_BADCODE", msg: "truncated instruction".into(), line })
    }

    fn push(&mut self, v: Value, line: u32) -> Result<(), VmError> {
        if self.stack.len() >= MAX_STACK {
            return Err(VmError {
                code: "E_KAZE_STACK",
                msg: "this expression nests too deeply; split it into smaller steps".to_string(),
                line,
            });
        }
        self.stack.push(v);
        Ok(())
    }

    fn pop(&mut self, line: u32) -> Result<Value, VmError> {
        self.stack.pop().ok_or_else(|| VmError {
            code: "E_KAZE_BADCODE",
            msg: "the value stack underflowed".to_string(),
            line,
        })
    }

    fn constant(&mut self, idx: usize, line: u32) -> Result<Value, VmError> {
        match self.module.consts.get(idx) {
            Some(Const::Num(n)) => Ok(Value::Num(*n)),
            Some(Const::Bool(b)) => Ok(Value::Bool(*b)),
            Some(Const::Str(s)) => {
                let text = s.clone();
                let r = self.arena.alloc(&text).map_err(|e| VmError {
                    code: "E_KAZE_MEMORY",
                    msg: e,
                    line,
                })?;
                Ok(Value::Str(r))
            }
            None => Err(VmError { code: "E_KAZE_BADCODE", msg: "missing constant".into(), line }),
        }
    }

    fn add(&mut self, a: Value, b: Value, line: u32) -> Result<Value, VmError> {
        match (a, b) {
            (Value::Num(x), Value::Num(y)) => {
                let sum = x + y;
                if !sum.is_finite() {
                    return Err(overflow(line));
                }
                Ok(Value::Num(sum))
            }
            (Value::Str(x), Value::Str(y)) => {
                let text = format!("{}{}", self.arena.get(x), self.arena.get(y));
                let r = self.arena.alloc(&text).map_err(|e| VmError {
                    code: "E_KAZE_MEMORY",
                    msg: e,
                    line,
                })?;
                Ok(Value::Str(r))
            }
            (Value::Str(x), other) => {
                let text = format!("{}{}", self.arena.get(x), caps::render_value(other, &self.arena, &self.heap));
                let r = self.arena.alloc(&text).map_err(|e| VmError {
                    code: "E_KAZE_MEMORY",
                    msg: e,
                    line,
                })?;
                Ok(Value::Str(r))
            }
            (other, Value::Str(y)) => {
                let text = format!("{}{}", caps::render_value(other, &self.arena, &self.heap), self.arena.get(y));
                let r = self.arena.alloc(&text).map_err(|e| VmError {
                    code: "E_KAZE_MEMORY",
                    msg: e,
                    line,
                })?;
                Ok(Value::Str(r))
            }
            (a, b) => Err(VmError {
                code: "E_KAZE_TYPE",
                msg: format!("cannot add {} to {}", a.type_name(), b.type_name()),
                line,
            }),
        }
    }

    fn arith(&mut self, op: Op, line: u32) -> Result<(), VmError> {
        let b = self.pop(line)?;
        let a = self.pop(line)?;
        let (x, y) = match (a, b) {
            (Value::Num(x), Value::Num(y)) => (x, y),
            (a, b) => {
                return Err(VmError {
                    code: "E_KAZE_TYPE",
                    msg: format!("cannot do arithmetic on {} and {}", a.type_name(), b.type_name()),
                    line,
                })
            }
        };
        let v = match op {
            Op::Sub => x - y,
            Op::Mul => x * y,
            Op::Div => {
                if y == 0.0 {
                    return Err(divide_by_zero(line));
                }
                x / y
            }
            Op::Mod => {
                if y == 0.0 {
                    return Err(divide_by_zero(line));
                }
                x % y
            }
            _ => unreachable!(),
        };
        if !v.is_finite() {
            return Err(overflow(line));
        }
        self.push(Value::Num(v), line)
    }

    fn compare(&self, op: Op, a: Value, b: Value, line: u32) -> Result<bool, VmError> {
        let ord = match (a, b) {
            (Value::Num(x), Value::Num(y)) => x.partial_cmp(&y),
            (Value::Str(x), Value::Str(y)) => self.arena.get(x).partial_cmp(self.arena.get(y)),
            (a, b) => {
                return Err(VmError {
                    code: "E_KAZE_TYPE",
                    msg: format!("cannot compare {} with {}", a.type_name(), b.type_name()),
                    line,
                })
            }
        };
        let ord = ord.ok_or_else(|| VmError {
            code: "E_KAZE_TYPE",
            msg: "this comparison has no answer (a value is not a number)".to_string(),
            line,
        })?;
        Ok(match op {
            Op::CmpLt => ord.is_lt(),
            Op::CmpLe => ord.is_le(),
            Op::CmpGt => ord.is_gt(),
            Op::CmpGe => ord.is_ge(),
            _ => unreachable!(),
        })
    }

    /// `record.field`. A missing field reads as `nothing` — an optional
    /// record field is normal data, not an error — but a field read on a
    /// non-record is a type error, because that is a bug in the script.
    fn field(&mut self, base: Value, field: Value, line: u32) -> Result<Value, VmError> {
        let name = match field {
            Value::Str(s) => self.arena.get(s).to_string(),
            other => {
                return Err(VmError {
                    code: "E_KAZE_TYPE",
                    msg: format!("a field name must be text, not {}", other.type_name()),
                    line,
                })
            }
        };
        match base {
            Value::Map(m) => {
                let key = self
                    .arena
                    .alloc(&name)
                    .map_err(|e| VmError { code: "E_KAZE_MEMORY", msg: e, line })?;
                Ok(self.heap.map_get(m, Value::Str(key), &self.arena).unwrap_or(Value::Unit))
            }
            other => Err(VmError {
                code: "E_KAZE_TYPE",
                msg: format!("'.{}' reads a field, but the value is {}", name, other.type_name()),
                line,
            }),
        }
    }

    fn length(&self, v: Value, line: u32) -> Result<f64, VmError> {
        match v {
            Value::Str(s) => Ok(self.arena.get(s).chars().count() as f64),
            Value::Arr(i) => Ok(self.heap.array(i).map(|a| a.len()).unwrap_or(0) as f64),
            Value::Map(i) => Ok(self.heap.map(i).map(|m| m.len()).unwrap_or(0) as f64),
            other => Err(VmError {
                code: "E_KAZE_TYPE",
                msg: format!("'length of' does not apply to {}", other.type_name()),
                line,
            }),
        }
    }

    fn index(&mut self, coll: Value, idx: Value, line: u32) -> Result<Value, VmError> {
        let i = match idx {
            Value::Num(n) => n,
            other => {
                return Err(VmError {
                    code: "E_KAZE_TYPE",
                    msg: format!("a position must be a number, not {}", other.type_name()),
                    line,
                })
            }
        };
        if i < 0.0 || i.fract() != 0.0 {
            return Err(VmError {
                code: "E_KAZE_BOUNDS",
                msg: format!("{} is not a valid position", fmt_num(i)),
                line,
            });
        }
        let i = i as usize;
        match coll {
            Value::Arr(a) => self
                .heap
                .array(a)
                .and_then(|items| items.get(i).copied())
                .ok_or_else(|| VmError {
                    code: "E_KAZE_BOUNDS",
                    msg: format!("position {} is past the end of the list", i),
                    line,
                }),
            Value::Str(s) => {
                let text: Vec<char> = self.arena.get(s).chars().collect();
                let Some(ch) = text.get(i).copied() else {
                    return Err(VmError {
                        code: "E_KAZE_BOUNDS",
                        msg: format!("position {} is past the end of the text", i),
                        line,
                    });
                };
                let one = ch.to_string();
                let r = self
                    .arena
                    .alloc(&one)
                    .map_err(|e| VmError { code: "E_KAZE_MEMORY", msg: e, line })?;
                Ok(Value::Str(r))
            }
            other => Err(VmError {
                code: "E_KAZE_TYPE",
                msg: format!("[ ] does not apply to {}", other.type_name()),
                line,
            }),
        }
    }

    fn call_cap(
        &mut self,
        cap_idx: usize,
        argc: usize,
        line: u32,
    ) -> Result<Option<CapabilityRequest>, VmError> {
        let cap = self
            .module
            .caps
            .get(cap_idx)
            .cloned()
            .ok_or_else(|| VmError { code: "E_KAZE_BADCODE", msg: "missing capability".into(), line })?;
        let mut args = vec![Value::Unit; argc];
        for i in (0..argc).rev() {
            args[i] = self.pop(line)?;
        }

        // The R7 gate runs for EVERY call, including the in-process families.
        crate::rt::hardware::authorise(
            &self.declared,
            &self.grants,
            self.policy.as_ref(),
            &cap.family,
            &cap.op,
        )
        .map_err(|d| VmError { code: d.code, msg: format!("{} ({})", d.reason, d.rule), line })?;
        if let Some(policy) = self.runtime_policy.as_ref() {
            crate::rt::hardware::authorise_runtime_policy(policy, &cap.family, &cap.op)
                .map_err(|d| VmError { code: d.code, msg: format!("{} ({})", d.reason, d.rule), line })?;
        }
        crate::rt::hardware::validate_arguments(&cap.family, &cap.op, &args, &self.arena)
            .map_err(|d| VmError { code: d.code, msg: format!("{} ({})", d.reason, d.rule), line })?;

        // Verdicts are control flow, not effects: the VM owns them, so a host
        // backend cannot fake an approval.
        if cap.family == "policy" && cap.op == "allow" {
            let v = args.first().copied().unwrap_or(Value::Unit);
            self.verdict = Some(Verdict::Allow(render(&v, &self.arena, &self.heap)));
            self.push(v, line)?;
            return Ok(None);
        }
        if cap.family == "policy" && cap.op == "reject" {
            let reason = match args.first() {
                Some(Value::Str(s)) => self.arena.get(*s).to_string(),
                other => other.map(|v| render(v, &self.arena, &self.heap)).unwrap_or_default(),
            };
            self.verdict = Some(Verdict::Reject(reason));
            self.push(Value::Unit, line)?;
            return Ok(None);
        }

        // Model and tool calls leave the VM as owned, bounded JSON requests.
        // Authorization and argument checks above run before any dispatcher
        // can observe the request. The synchronous host never fabricates a
        // replacement result.
        if matches!((cap.family.as_str(), cap.op.as_str()),
            ("ai", "prompt" | "complete" | "classify") | ("tool", "call" | "run"))
        {
            let mut nodes = 0usize;
            let serialized_args = args.iter()
                .map(|value| value_to_json(*value, &self.arena, &self.heap, 0, &mut nodes, line))
                .collect::<Result<Vec<_>, _>>()?;
            let json = serde_json::Value::Array(serialized_args);
            let bytes = serde_json::to_vec(&json)
                .map_err(|_| VmError { code: "E_KAZE_HOST", msg: "capability arguments are not valid JSON".into(), line })?;
            if bytes.len() > MAX_CAPABILITY_JSON_BYTES {
                return Err(VmError { code: "E_KAZE_MEMORY", msg: "capability arguments exceeded the 1 MiB limit".into(), line });
            }
            self.host.check_async_dispatch().map_err(|message| {
                let (code, msg) = split_host_code(message);
                VmError { code, msg, line }
            })?;
            let id = NEXT_CAPABILITY_REQUEST_ID
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .map_err(|_| VmError { code: "E_KAZE_STATE", msg: "capability request identifiers exhausted".into(), line })?;
            return Ok(Some(CapabilityRequest {
                id,
                family: cap.family,
                op: cap.op,
                args: json,
                line,
            }));
        }

        let result = self
            .host
            .call(&cap.family, &cap.op, &args, &mut self.arena, &mut self.heap)
            .map_err(|msg| {
                // A live backend may name its own code (`E_KAZE_LINK_TIMEOUT: …`),
                // so the report says what actually went wrong instead of a
                // generic host failure. Only codes the engine knows survive;
                // anything else stays `E_KAZE_HOST`.
                let (code, msg) = split_host_code(msg);
                VmError { code, msg, line }
            })?;
        self.push(result, line)?;
        Ok(None)
    }

}

fn value_to_json(
    value: Value,
    arena: &Arena,
    heap: &Heap,
    depth: usize,
    nodes: &mut usize,
    line: u32,
) -> Result<serde_json::Value, VmError> {
    *nodes = nodes.saturating_add(1);
    if depth > 32 || *nodes > 16_384 {
        return Err(VmError { code: "E_KAZE_MEMORY", msg: "capability arguments are too deeply nested or complex".into(), line });
    }
    use serde_json::{Map, Number, Value as Json};
    match value {
        Value::Unit => Ok(Json::Null),
        Value::Bool(value) => Ok(Json::Bool(value)),
        Value::Num(value) => Number::from_f64(value).map(Json::Number)
            .ok_or_else(|| VmError { code: "E_KAZE_TYPE", msg: "capability arguments must contain finite numbers".into(), line }),
        Value::Str(value) => Ok(Json::String(arena.get(value).to_string())),
        Value::Arr(index) => {
            let values = heap.array(index)
                .ok_or_else(|| VmError { code: "E_KAZE_BADCODE", msg: "capability argument contains an invalid list".into(), line })?;
            values.iter().map(|value| value_to_json(*value, arena, heap, depth + 1, nodes, line))
                .collect::<Result<Vec<_>, _>>().map(Json::Array)
        }
        Value::Map(index) => {
            let values = heap.map(index)
                .ok_or_else(|| VmError { code: "E_KAZE_BADCODE", msg: "capability argument contains an invalid record".into(), line })?;
            let mut output = Map::new();
            for (key, value) in values {
                let Value::Str(key) = key else {
                    return Err(VmError { code: "E_KAZE_TYPE", msg: "capability record keys must be text".into(), line });
                };
                let key = arena.get(*key).to_string();
                if output.contains_key(&key) {
                    return Err(VmError { code: "E_KAZE_TYPE", msg: "capability record has duplicate keys".into(), line });
                }
                output.insert(key, value_to_json(*value, arena, heap, depth + 1, nodes, line)?);
            }
            Ok(Json::Object(output))
        }
    }
}

/// Codes a capability host may raise. The list is closed on purpose: a host
/// cannot invent an error the CLI has never been taught to explain.
const HOST_CODES: &[&str] = &[
    "E_CAP_MISSING",
    "E_CAP_DENIED",
    "E_KAZE_MEMORY",
    "E_KAZE_RANGE",
    "E_KAZE_LINK",
    "E_KAZE_LINK_TIMEOUT",
    "E_COMMAND_REJECTED",
    "E_ARMING_DENIED",
    "E_SAFETY_VIOLATION",
];

fn split_host_code(msg: String) -> (&'static str, String) {
    if let Some((head, rest)) = msg.split_once(':') {
        for code in HOST_CODES {
            if head == *code {
                return (code, rest.trim_start().to_string());
            }
        }
    }
    ("E_KAZE_HOST", msg)
}

fn jump(pc: usize, imm: i32, len: usize, line: u32) -> Result<usize, VmError> {
    let target = pc as i64 + imm as i64;
    if target < 0 || target as usize > len {
        return Err(VmError { code: "E_KAZE_BADCODE", msg: "a jump left the code".into(), line });
    }
    Ok(target as usize)
}

fn overflow(line: u32) -> VmError {
    VmError {
        code: "E_KAZE_OVERFLOW",
        msg: "this calculation overflowed; the run stopped instead of continuing with a wrong \
              number"
            .to_string(),
        line,
    }
}

fn divide_by_zero(line: u32) -> VmError {
    VmError { code: "E_KAZE_DIV0", msg: "dividing by zero".to_string(), line }
}

fn render(v: &Value, arena: &Arena, heap: &Heap) -> String {
    crate::sfs::caps::render_value(*v, arena, heap)
}

#[cfg(test)]
mod continuation_tests {
    use super::*;
    use crate::sfs::caps::{Action, RuntimeHost, SimState};
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

    struct CountingHost(Arc<AtomicUsize>);

    impl CapHost for CountingHost {
        fn call(&mut self, family: &str, op: &str, _args: &[Value], _arena: &mut Arena, _heap: &mut Heap) -> Result<Value, String> {
            if family == "log" && op == "info" {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(Value::Unit)
            } else {
                Err(format!("E_CAP_MISSING: no test backend for {family}.{op}"))
            }
        }
        fn actions(&self) -> &[Action] { &[] }
        fn logs(&self) -> &[(String, String)] { &[] }
    }

    fn host() -> RuntimeHost {
        RuntimeHost::new(SimState::default(), Vec::new(), Vec::new())
    }

    #[test]
    fn slices_preserve_control_flow_and_do_not_replay_real_log_calls() {
        let source = "needs: log\naction main():\n    let n = 0\n    while n < 3:\n        log \"tick\"\n        set n to n + 1\n    if n == 3:\n        return n\n    otherwise:\n        return 0\n";
        let module = crate::sfs::prepare(source, "<continuation>").unwrap().module;
        let item = module.item("main").unwrap();
        let grants = vec!["log".to_string()];
        let policy = serde_json::json!({"tools": {"allow": ["kaze.log.*"]}});

        let mut synchronous_host = host();
        let (expected_value, expected_instructions) = {
            let mut vm = Vm::new(&module, &mut synchronous_host, &module.needs, &grants, Some(&policy), Limits::default());
            (vm.run_item(item, &[]).unwrap(), vm.instructions)
        };
        let mut sliced_host = host();
        let mut vm = Vm::new(&module, &mut sliced_host, &module.needs, &grants, Some(&policy), Limits::default());
        let mut state = vm.start_item(item, &[]).unwrap();
        let mut yields = 0;
        let actual_value = loop {
            match vm.run_slice(&mut state, 1).unwrap() {
                SliceOutcome::Yielded => yields += 1,
                SliceOutcome::Capability(request) => panic!("unexpected request: {request:?}"),
                SliceOutcome::Complete(value) => break value,
            }
        };
        assert!(yields > 3);
        assert_eq!(actual_value, expected_value);
        assert_eq!(actual_value, Value::Num(3.0));
        assert_eq!(vm.instructions, expected_instructions);
        assert_eq!(vm.run_slice(&mut state, 1).unwrap_err().code, "E_KAZE_STATE");
        drop(vm);
        assert_eq!(sliced_host.logs(), synchronous_host.logs());
        assert_eq!(sliced_host.logs().len(), 3);
        assert_eq!(sliced_host.actions().iter().map(|a| a.render()).collect::<Vec<_>>(),
                   synchronous_host.actions().iter().map(|a| a.render()).collect::<Vec<_>>());
    }

    #[test]
    fn continuation_cannot_be_reused_or_moved_between_vms() {
        let module = crate::sfs::prepare("action main():\n    return 7\n", "<continuation>").unwrap().module;
        let item = module.item("main").unwrap();
        let mut first_host = host();
        let mut second_host = host();
        let mut first = Vm::new(&module, &mut first_host, &module.needs, &[], None, Limits::default());
        let mut second = Vm::new(&module, &mut second_host, &module.needs, &[], None, Limits::default());
        let mut state = first.start_item(item, &[]).unwrap();
        assert_eq!(second.run_slice(&mut state, 1).unwrap_err().code, "E_KAZE_STATE");
        assert_eq!(first.start_item(item, &[]).err().unwrap().code, "E_KAZE_STATE");
        assert_eq!(first.run_slice(&mut state, 0).unwrap_err().code, "E_KAZE_STATE");
        assert_eq!(first.run_slice(&mut state, 1).unwrap(), SliceOutcome::Yielded);
        first.cancel_item(&mut state).unwrap();
        assert_eq!(first.run_slice(&mut state, 1).unwrap_err().code, "E_KAZE_STATE");
        assert_eq!(first.run_item(item, &[]).unwrap(), Value::Num(7.0));
    }

    #[test]
    fn denial_and_fuel_failure_are_terminal_across_slices() {
        let module = crate::sfs::prepare("needs: log\naction main():\n    log \"blocked\"\n", "<continuation>").unwrap().module;
        let mut denied_host = host();
        let mut vm = Vm::new(&module, &mut denied_host, &module.needs, &[], None, Limits::default());
        let mut state = vm.start_item(module.item("main").unwrap(), &[]).unwrap();
        let error = loop {
            match vm.run_slice(&mut state, 1) {
                Ok(SliceOutcome::Yielded) => (),
                Err(error) => break error,
                Ok(SliceOutcome::Complete(_) | SliceOutcome::Capability(_)) => panic!("denied operation succeeded"),
            }
        };
        assert_eq!(error.code, "E_CAP_MISSING");
        assert_eq!(error.line, 3);
        assert_eq!(vm.run_slice(&mut state, 1).unwrap_err().code, "E_KAZE_STATE");
        drop(vm);
        assert!(denied_host.actions().is_empty());
        assert!(denied_host.logs().is_empty());

        let loop_module = crate::sfs::prepare("action main():\n    while true:\n        let n = 1\n", "<continuation>").unwrap().module;
        let mut loop_host = host();
        let mut loop_vm = Vm::new(&loop_module, &mut loop_host, &loop_module.needs, &[], None, Limits { fuel: 8, ..Limits::default() });
        let mut state = loop_vm.start_item(loop_module.item("main").unwrap(), &[]).unwrap();
        let error = loop {
            match loop_vm.run_slice(&mut state, 2) {
                Ok(SliceOutcome::Yielded) => (),
                Err(error) => break error,
                Ok(SliceOutcome::Complete(_) | SliceOutcome::Capability(_)) => panic!("runaway loop completed"),
            }
        };
        assert_eq!(error.code, "E_TIMEOUT");
        assert_eq!(loop_vm.run_slice(&mut state, 1).unwrap_err().code, "E_KAZE_STATE");
    }

    #[test]
    fn authorized_ai_call_yields_owned_request_without_fabricating_a_result() {
        let source = "needs: log, ai\naction main():\n    log \"one side effect\"\n    return ai.prompt(\"fresh marker\")\n";
        let module = crate::sfs::prepare(source, "<async-call>").unwrap().module;
        let item = module.item("main").unwrap().clone();
        let grants = vec!["log".to_string(), "ai".to_string()];
        let policy = serde_json::json!({"tools": {"allow": ["kaze.*"]}});
        let count = Arc::new(AtomicUsize::new(0));
        let mut vm = Vm::new_owned(
            module.clone(),
            Box::new(CountingHost(Arc::clone(&count))),
            module.needs.clone(),
            grants,
            Some(policy.clone()),
            Limits::default(),
        );
        vm.set_runtime_policy(Some(&policy));
        let mut state = vm.start_item(&item, &[]).unwrap();
        let request = loop {
            match vm.run_slice(&mut state, 1).unwrap() {
                SliceOutcome::Yielded => (),
                SliceOutcome::Capability(request) => break request,
                SliceOutcome::Complete(_) => panic!("AI request unexpectedly completed"),
            }
        };
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(request.family, "ai");
        assert_eq!(request.op, "prompt");
        assert_eq!(request.args, serde_json::json!(["fresh marker"]));
        assert_eq!(request.line, 4);
        assert_eq!(vm.run_slice(&mut state, 1).unwrap_err().code, "E_KAZE_STATE");
        assert_eq!(vm.resume_capability(&mut state, request.id + 1, Ok(serde_json::json!("must not be accepted"))).unwrap_err().code, "E_KAZE_STATE");
        assert_eq!(vm.run_slice(&mut state, 1).unwrap_err().code, "E_KAZE_STATE");
        vm.cancel_item(&mut state).unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 1, "earlier effect executed exactly once");
    }

    #[test]
    fn denied_async_call_never_becomes_a_dispatch_request() {
        let source = "needs: log, ai\naction main():\n    log \"one side effect\"\n    return ai.prompt(\"private prompt\")\n";
        let module = crate::sfs::prepare(source, "<async-denial>").unwrap().module;
        let item = module.item("main").unwrap().clone();
        let grants = vec!["log".to_string(), "ai".to_string()];
        let policy = serde_json::json!({"tools": {"deny": ["kaze.ai.*"]}});
        let count = Arc::new(AtomicUsize::new(0));
        let mut vm = Vm::new_owned(
            module.clone(),
            Box::new(CountingHost(Arc::clone(&count))),
            module.needs.clone(), grants, Some(policy.clone()), Limits::default(),
        );
        vm.set_runtime_policy(Some(&serde_json::json!({"tools": {"allow": ["kaze.*"]}})));
        let mut state = vm.start_item(&item, &[]).unwrap();
        let error = loop {
            match vm.run_slice(&mut state, 2) {
                Ok(SliceOutcome::Yielded) => (),
                Err(error) => break error,
                Ok(SliceOutcome::Capability(_)) => panic!("denied request reached the dispatcher"),
                Ok(SliceOutcome::Complete(_)) => panic!("denied request returned success"),
            }
        };
        assert_eq!(error.code, "E_CAP_DENIED");
        assert_eq!(error.line, 4);
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(vm.run_slice(&mut state, 1).unwrap_err().code, "E_KAZE_STATE");
    }

    #[test]
    fn uncertain_dispatched_effect_is_recorded_as_unknown_and_is_terminal() {
        let source = "needs: tool\naction main():\n    return tool.call(\"write_once\")\n";
        let module = crate::sfs::prepare(source, "<async-unknown>").unwrap().module;
        let item = module.item("main").unwrap().clone();
        let grants = vec!["tool".to_string()];
        let policy = serde_json::json!({"tools": {"allow": ["kaze.*"]}});
        let mut vm = Vm::new_owned(
            module.clone(),
            Box::new(host()),
            module.needs.clone(),
            grants,
            Some(policy.clone()),
            Limits::default(),
        );
        vm.set_runtime_policy(Some(&policy));
        let mut state = vm.start_item(&item, &[]).unwrap();
        let request = loop {
            match vm.run_slice(&mut state, 8).unwrap() {
                SliceOutcome::Yielded => (),
                SliceOutcome::Capability(request) => break request,
                SliceOutcome::Complete(_) => panic!("tool request unexpectedly completed"),
            }
        };
        let error = vm.resume_capability(&mut state, request.id, Err(VmError {
            code: "E_KAZE_OUTCOME_UNKNOWN",
            msg: "no tool response was received".into(),
            line: 0,
        })).unwrap_err();
        assert_eq!(error.code, "E_KAZE_OUTCOME_UNKNOWN");
        assert_eq!(error.line, request.line);
        assert_eq!(vm.actions().len(), 1);
        assert_eq!(vm.actions()[0].result, "<unknown outcome>");
        assert_eq!(vm.run_slice(&mut state, 1).unwrap_err().code, "E_KAZE_STATE");
    }
}
